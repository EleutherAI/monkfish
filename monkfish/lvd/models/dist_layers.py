import os
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shrd
import jax.lax as lax
import equinox as eqx

import monkfish.lvd.models.dist_utils as du

#TODO: Subclass eqx.Module properly

def make_f_dict(pre_dict, dist_manager):
    f_dict = {}
    for name, (p_spec, path) in pre_dict.items():
        partition_spec = shrd.PartitionSpec(*p_spec)
        sharding = dist_manager.sharding(partition_spec)
        path_f = lambda a, b: os.path.join(b, a)

        f_dict[name] = {
            "sharding": sharding,
            "path_fn": functools.partial(path_f, path)
        }
    return f_dict

class ShrdMHAttention(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    q: jax.Array
    k: jax.Array
    v: jax.Array
    o: jax.Array
    qk_layer_norm: jax.Array | None
    theta_factor: jax.Array
    scale: jax.Array = eqx.field(static=True)
    
    def _f_dict(self):
        pre_dict = {
            "q": ((("mp","fsdp"), None, None), "q.pkl"),
            "k": ((("mp","fsdp"), None, None), "k.pkl"),
            "v": ((("mp","fsdp"), None, None), "v.pkl"),
            "o": ((("mp","fsdp"), None, None), "o.pkl"),
            "qk_layer_norm": ((None,), "qk_layer_norm.pkl"),
            "theta_factor":((), "theta_factor.pkl"),
            "scale":((), "scale.pkl")
        }
        return make_f_dict(pre_dict, self.dist_manager)
    
    def __init__(self, dist_manager, key, d_model, 
                 n_head, d_qk, d_v, qk_layer_norm=False, theta_factor=10000):
        keys = jax.random.split(key, 6)
    
        self.scale = 1/jnp.sqrt(n_head*d_v*d_model)

        self.dist_manager = dist_manager

        #Init q 
        shape = (n_head, d_model, d_qk)
        std = 1
        self.q = self.dist_manager.init_randn_array(
            shape, std, self._f_dict()["q"]["sharding"], keys[0])
        
        #Init k
        shape = (n_head, d_model, d_qk)
        std = 1
        self.k = self.dist_manager.init_randn_array(
            shape, std, self._f_dict()["k"]["sharding"], keys[1])
        
        #Init v
        shape = (n_head, d_model, d_v)
        std = 1
        self.v = self.dist_manager.init_randn_array(
            shape, std, self._f_dict()["v"]["sharding"], keys[2])
        
        #Init o
        shape = (n_head, d_v, d_model)
        std = 1
        self.o = self.dist_manager.init_randn_array(
            shape, std, self._f_dict()["o"]["sharding"], keys[3])
        
        #Init theta_factor 
        shape = ()
        std = 0
        self.theta_factor = self.dist_manager.init_randn_array(
            shape, std, self._f_dict()["theta_factor"]["sharding"], keys[4])
        self.theta_factor += theta_factor
        
        #Init qk_layer_norm 
        if qk_layer_norm:
            shape = (n_head,)
            std = 0
            self.qk_layer_norm = self.dist_manager.init_randn_array(
                shape, std, self._f_dict()["qk_layer_norm"]["sharding"], keys[5])
        else:
            self.qk_layer_norm = None
    
    def _mha(self, x, q, k, v, o, mask):
        par_sha = jax.vmap(self._sha, in_axes=(None, 0, 0, 0, None))

        #[pos x d_model] x [head x d_model x d_qk] x 
        #[head x d_model x d_qk] x [head x d_model x d_v] -> 
        #[head x pos x d_v]
        y = par_sha(x, q, k, v, mask)
        
        #[head x pos x d_v] x [head x d_v x d_model] -> [pos x d_model]
        z = jnp.einsum("ijk,ikl->jl", y, o)
        return z
        
    def _sha(self, x,  q, k, v, mask):
        
        #[pos x d_model] x [d_model x d_qk] -> [pos x d_qk]
        pre_qs = jnp.einsum("ij,jk->ik", x, q)
        rot_qs = self._rope_embed(pre_qs)
        
        #[pos x d_model] x [d_model x d_qk] -> [pos x d_qk]
        pre_ks = jnp.einsum("ij,jk->ik", x, k)
        rot_ks = self._rope_embed(pre_ks)
        
        #[pos x d_model] x [d_model x d_v] -> [pos x d_v]
        vs = jnp.einsum("ij,jk->ik", x, v)

        #[pos x d_qk] x [pos x d_qk] -> [pos x pos]
        unmasked_attention = jnp.einsum("ik,jk->ij", rot_qs, rot_ks)
        masked_attention = unmasked_attention+mask
        attention_weights = jax.nn.softmax(masked_attention)

        #[pos x pos] x [pos x d_v] -> [pos x d_model]
        y = jnp.einsum("ij,jk->ik", attention_weights, vs)
        
        return y
    #[pos x d_qk] -> [pos x d_qk]
    def _rope_embed(self, x):
        d_qk = self.q.shape[2]
        d_pos = x.shape[0]
        d_rope = x.shape[1] // 2
        rate_vector = self.theta_factor*(-jnp.arange(0,d_rope)/d_rope)
        pos_vector = jnp.arange(0, d_pos)

        rot_factor = jnp.einsum("i,j->ij", pos_vector, rate_vector)
        sin_factor, cos_factor = jnp.sin(rot_factor),jnp.cos(rot_factor)

        x1,x2 = x[:,:d_rope],x[:,d_rope:]

        y1 = cos_factor*x1 - sin_factor*x2
        y2 = sin_factor*x1 + cos_factor*x2

        y = jnp.concatenate([y1, y2], axis=1)
        
        #Norm step
        if self.qk_layer_norm is None:
            y = y/(d_qk**(1/4))
        else:
            raise NotImplementedError
        
        return y

    def _causal_mask(self, size):
        # Creating a lower triangular matrix of ones (including diagonal)
        mask = jnp.tril(jnp.ones((size, size)))
        mask = mask - 1
        #TODO: Less hacky
        mask = mask * 1000000000
        return mask

    #[pos x d_model] -> [pos x d_model]
    def __call__(self, x):
        seq_length = x.shape[0]  # Get the sequence length
        causal_mask = self._causal_mask(seq_length)  # Create the causal mask for the sequence
        y = self._mha(x, self.q, self.k, self.v, self.o, causal_mask)*self.scale
        return y

#TODO: Support dilation
class ShrdConv(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    kernel: jax.Array
    bias: jax.Array
    scale: jax.Array = eqx.field(static=True)
    padding: str = eqx.field(static=True)
    
    def _f_dict(self):
        pre_dict = {
            "kernel": ((("mp","fsdp"), None, None, None), "kernel.pkl"),
            "bias": ((None,), "bias.pkl"),
            "scale":((), "scale.pkl")
        }
        return make_f_dict(pre_dict, self.dist_manager)
    
    def __init__(self, dist_manager, key, h, w, in_dim, out_dim, 
                 padding="SAME", bias=False):
        key1,key2 = jax.random.split(key)
        self.dist_manager = dist_manager
        
        self.scale = 1/jax.numpy.sqrt(h*w*in_dim)

        self.padding = padding
        
        #Init kernel 
        shape = (out_dim, in_dim, h, w)
        std = 1
        self.kernel = self.dist_manager.init_randn_array(
            shape, std, self._f_dict()["kernel"]["sharding"], key1)
        
        #Init bias
        if bias:
            shape = (out_dim,)
            std = 0
            self.bias = self.dist_manager.init_randn_array(
                shape, std, self._f_dict()["bias"]["sharding"], key2)
        else:
            self.bias = None
    
    #Assumping padding = SAME
    #[in_dim x height x width] -> [out_dim x height x width]
    def __call__(self, x):
        y = lax.conv_with_general_padding(
            x[jnp.newaxis,:,:], self.kernel, 
            window_strides=(1,1), padding=self.padding, 
            lhs_dilation=None, rhs_dilation=None)[0,:,:,:]
        y = y*self.scale
        if self.bias is not None:
            y = y + self.bias
        return y

class ShrdLinear(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    weight: jax.Array
    bias: jax.Array | None
    scale: jax.Array = eqx.field(static=True)
    
    def _f_dict(self):
        pre_dict = {
            "weight": ((("mp","fsdp"), None), "weight.pkl"),
            "bias": (((None),), "bias.pkl"),
            "scale":((), "scale.pkl")
        }
        return make_f_dict(pre_dict, self.dist_manager)
    
    def __init__(self, dist_manager, key, in_dim, out_dim, bias=False):
        self.dist_manager = dist_manager
        key1,key2 = jax.random.split(key)

        self.scale = 1/jnp.sqrt(in_dim)

        #Init weight
        shape = (in_dim, out_dim)
        std = 1
        self.weight = self.dist_manager.init_randn_array(
            shape, std, self._f_dict()["weight"]["sharding"], key1)
        
        #Init bias
        if bias:
            shape = (out_dim,)
            std = 0
            self.bias = self.dist_manager.init_randn_array(
                shape, std, self._f_dict()["bias"]["sharding"], key2)
        else:
            self.bias = None
    
    #[h] -> [h]
    def __call__(self, x):
        y = jnp.einsum("i,ij->j",x, self.weight)*self.scale
        if self.bias is not None:
            y = y + self.bias
        return y

class ConvResBlock(eqx.Module):
    layer1: ShrdConv
    layer2: ShrdConv

    def __init__(self, dist_manager, key,  in_dim, latent_dim):
        key1, key2 = jax.random.split(key)

        self.layer1 = ShrdConv(dist_manager, key1, 3, 3, in_dim, latent_dim)
        self.layer2 = ShrdConv(dist_manager, key2, 3, 3, latent_dim, in_dim)

    #[in_dim x height x width] -> [in_dim x height x width]
    def __call__(self, x):
        h = self._norm(x)
        h = self.layer1(h)
        h = self.layer2(h)
        y = x + h
        return y
    
    def _norm(self, x):
        m = jnp.mean(x, axis=2)
        s = jnp.std(x, axis=2)
        y = (x-m[:,:, jnp.newaxis])/(s[:,:, jnp.newaxis])
        return y

class TransformerBlock(eqx.Module):
    mlpl1: ShrdLinear
    mlpl2: ShrdLinear
    attn: ShrdMHAttention

    def __init__(self, dist_manager, key, res_dim, mlp_dim, 
                 qk_dim, v_dim, n_head):
        self.mlpl1 = ShrdLinear(dist_manager, key, res_dim, mlp_dim)
        self.mlpl2 = ShrdLinear(dist_manager, key, mlp_dim, res_dim)
        self.attn = ShrdMHAttention(dist_manager, key, res_dim, n_head, qk_dim, v_dim)
    
    def _norm(self, x):
        m = jnp.mean(x, axis=1)
        s = jnp.std(x, axis=1)
        y = (x-m[:, jnp.newaxis])/(s[:, jnp.newaxis])
        return y
    
    #[pos x res_dim] -> [pos x res_dim]
    def __call__(self, x):
        h1 = self._norm(x)
        h2 = jax.vmap(self.mlpl1)(h1)
        h3 = jax.vmap(self.mlpl2)(h2)
        h4 = self.attn(h1)
        y = (h3 + h4)#/jnp.sqrt(2)
        return y
        




