import os
import functools
import dataclasses

import jax
import jax.numpy as jnp
import jax.sharding as shrd
import jax.lax as lax
import equinox as eqx

import catfish.lvd.models.dist_utils as du

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
    theta_factor: jax.Array = eqx.field(static=True)
    
    def _f_dict(self):
        pre_dict = {
            "q": ((("mp","fsdp"), None, None), "q.pkl"),
            "k": ((("mp","fsdp"), None, None), "k.pkl"),
            "v": ((("mp","fsdp"), None, None), "v.pkl"),
            "o": ((("mp","fsdp"), None, None), "o.pkl"),
            "qk_layer_norm": (((None,),), "qk_layer_norm.pkl"),
            "theta_factor":((), "theta_factor.pkl")
        }
        return make_f_dict(pre_dict, self.dist_manager)
    
    def __init__(self, dist_manager, key, d_model, n_head, d_qk, d_v, qk_layer_norm=False, theta_factor=10000):
        self.dist_manager = dist_manager

        #Init q 
        shape = (n_head, d_model, d_qk)
        std = 1
        self.q = self.dist_manager.init_randn_array(
            self, shape, std, self._f_dict()["q"]["sharding"])
        
        #Init k
        shape = (n_head, d_model, d_qk)
        std = 1
        self.k = self.dist_manager.init_randn_array(
            self, shape, std, self._f_dict()["k"]["sharding"])
        
        #Init v
        shape = (n_head, d_model, d_v)
        std = 1
        self.v = self.dist_manager.init_randn_array(
            self, shape, std, self._f_dict()["v"]["sharding"])
        
        #Init o
        shape = (n_head, d_v, d_model)
        std = 1
        self.o = self.dist_manager.init_randn_array(
            self, shape, std, self._f_dict()["v"]["sharding"])
        
        #Init theta_factor 
        shape = ()
        std = 0
        self.theta_factor = self.dist_manager.init_randn_array(
            self, shape, std, self._f_dict()["v"]["sharding"])
        self.theta_factor += theta_factor
        
        #Init qk_layer_norm 
        if qk_layer_norm:
            shape = ()
            std = 0
            self.qk_layer_norm = self.dist_manager.init_randn_array(
                self, shape, std, self._f_dict()["qk_layer_norm"]["sharding"])
        else:
            self.qk_layer_norm = None
    
    def _mha(self, x, q, k, v, o, mask=None):
        par_sha = jax.vmap(self._sha, in_axes=(None, 0, 0, 0, None))

        #[pos x d_model] x [head x d_model x d_qk] x 
        #[head x d_model x d_qk] x [head x d_model x d_v] -> 
        #[head x pos x d_v]
        y = par_sha(x, q, k, v)
        
        #[head x pos x d_v] x [head x d_v x d_model] -> [pos x d_model]
        z = jnp.einsum("ijk,ikl->jl", y, o)
        return z
        
    def _sha(self, x,  q, k, v, mask=None):
        
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
        masked_attention = unmasked_attention-mask
        attention_weights = jax.nn.softmax(masked_attention)

        #[pos x pos] x [pos x d_v] -> [pos x d_model]
        y = jnp.einsum("ij,jk->ik", attention_weights, vs)
        
        return y
    
    def _rope_embed(self, x):
        d_qk = self.q.shape[2]
        d_pos = x.shape[0]
        d_rope = x.shape[1] // 2
        rate_vector = self.theta_factor*(-jnp.arange(0,d_rope)/d_rope)
        pos_vector = jnp.arange(0, d_pos)

        rot_factor = jnp.einsum("i,j->ij", pos_vector, rate_vector)
        sin_factor, cos_factor = jnp.sin(rot_factor),jnp.cos(rot_factor)

        x1,x2 = x[:,:d_pos],x[:,d_pos:]

        y1 = cos_factor*x1 - sin_factor*x2
        y2 = sin_factor*x1 + cos_factor*x2

        y = jax.concatenate([y1, y2], axis=1)
        
        #Norm step
        if self.qk_layer_norm is None:
            y = y/(d_qk^(1/4))
        else:
            raise NotImplementedError
        
        return y

    def _causal_mask(self, size):
        # Creating a lower triangular matrix of ones (including diagonal)
        mask = jnp.tril(jnp.ones((size, size)))
        mask = mask - 1
        mask = mask * jnp.INF
        return mask
    
    #[pos x d_model] -> [pos x d_model]
    def __call__(self, x):
        seq_length = x.shape[0]  # Get the sequence length
        causal_mask = self._causal_mask(seq_length)  # Create the causal mask for the sequence
        y = self._mha(x, self.q, self.k, self.v, self.o, causal_mask)
        return y
    
    def save(self, path_prefix):
        for key, value in self._f_dict().items():
            array = getattr(self, key)
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"](path_prefix)
            self.dist_manager.save_array(
                array, sharding, path)
    
    def load(self, path_prefix):
        new_self = self
        for key, value in self._f_dict().items():
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"]
            array = self.dist_manager.load_array(
                sharding, path)
            
            where = lambda x: getattr(x, key)
            new_self = eqx.tree_at(where, new_self, array)
        
        return new_self

#TODO: Support dilation
class ShrdConv(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    kernel: jax.Array
    bias: jax.Array
    scale: jax.Array = eqx.field(static=True)
    padding: str = eqx.field(static=True)
    
    def _f_dict(self):
        pre_dict = {
            "kernel": ((None,None,("mp","fsdp"), None), "weight.pkl"),
            "bias": ((None), "bias.pkl"),
            "scale":((), "scale.pkl")
        }
        return make_f_dict(pre_dict, self.dist_manager)
    
    def __init__(self, dist_manager, key, h, w, in_dim, out_dim, 
                 padding="SAME", bias=False):
        self.dist_manager = dist_manager
        
        self.scale = 1/jax.numpy.sqrt(h*w*in_dim)

        self.padding = padding
        
        #Init kernel 
        shape = (h, w, in_dim, out_dim)
        std = 1
        self.weight = self.dist_manager.init_randn_array(
            self, shape, std, self._f_dict()["weight"]["sharding"])
        
        #Init bias
        if bias:
            shape = (out_dim,)
            std = 0
            self.bias = self.dist_manager.init_randn_array(
                self, shape, std, self._f_dict()["bias"]["sharding"])
        else:
            self.bias = None
    
    #Assumping padding = SAME
    #[in_dim x height x width] -> [out_dim x width x in_dim]
    def __call__(self, x):
        y = lax.conv_with_general_padding(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y
    
    def save(self, path_prefix):
        for key, value in self._f_dict().items():
            array = getattr(self, key)
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"]
            self.dist_manager.save_array(
                array, sharding, path)
    
    def load(self, path_prefix):
        new_self = self
        for key, value in self._f_dict().items():
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"]
            array = self.dist_manager.load_array(
                sharding, path)
            
            where = lambda x: getattr(x, key)
            new_self = eqx.tree_at(where, new_self, array)
        
        return new_self

class ShrdLinear(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    weight: jax.Array
    bias: jax.Array | None
    scale: jax.Array
    
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

        self.scale = 1/jnp.sqrt(out_dim)

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
    
    def save(self, path_prefix):
        for key, value in self._f_dict().items():
            array = getattr(self, key)
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"]
            print(key, path, sharding)
            self.dist_manager.save_array(
                array, sharding, path)
    
    def load(self, path_prefix):
        new_self = self
        for key, value in self._f_dict().items():
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"]
            array = self.dist_manager.load_array(
                sharding, path)
            
            where = lambda x: getattr(x, key)
            new_self = eqx.tree_at(where, new_self, array)
        
        return new_self

class ConvResBlock(eqx.Module):
    layer1: ShrdConv
    layer2: ShrdConv

    def __init__(self, dist_manager, key,  in_dim, latent_dim):
        key1, key2 = jax.random.split(key)

        self.dist_manager = dist_manager

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
        m = jnp.mean(x, dim=0)
        s = jnp.std(x, dim=0)
        y = (x-m)/(s)
        return y

    def save(self, path_prefix):
        path1 = os.path.join(path_prefix, f"layer_1")
        self.layer1.save(path1)
        path2 = os.path.join(path_prefix, f"layer_2")
        self.layer2.save(path2)
    
    def load(self, path_prefix):
        new_self = self

        path1 = os.path.join(path_prefix, f"layer_1")
        layer1 = self.layer1.load(path1)
        where = lambda x: x.layer1
        new_self = eqx.tree_at(where, new_self, layer1)
        
        path2 = os.path.join(path_prefix, f"layer_2")
        layer2 = self.layer2.load(path2)
        where = lambda x: x.layer2
        new_self = eqx.tree_at(where, new_self, layer2)

        return new_self

class TransformerBlock(eqx.Module):
    mlpl1: ShrdLinear
    mlpl2: ShrdLinear
    attn: ShrdMHAttention

    def __init__(self, dist_manager, key, res_dim, latent_dim, qk_dim, v_dim, n_head):
        self.mlpl1 = ShrdLinear(dist_manager, key, res_dim, latent_dim)
        self.mlpl2 = ShrdLinear(dist_manager, key, latent_dim, res_dim)
        self.attn = ShrdMHAttention(dist_manager, key, res_dim, n_head, qk_dim, v_dim)
    
    def _norm(self, x):
        m = jnp.mean(x, dim=1)
        s = jnp.std(x, dim=1)
        y = (x-m)/(s)
        return y
    
    #[pos x res_dim] -> [pos x res_dim]
    def __call__(self, x):
        h1 = self._norm(x)
        h2 = jax.vmap(x)(self.mpll1)(h1)
        h3 = jax.vmap(x)(self.mpll2)(h2)
        h4 = self.attn(h1)
        y = h3 + h4 + x
        return y
    
    def save(self, path_prefix):
        path1 = os.path.join(path_prefix, f"mlpl_1")
        self.mlpl1.save(path1)
        path2 = os.path.join(path_prefix, f"mlpl_2")
        self.mlpl2.save(path2)
        path3 = os.path.join(path_prefix, f"attn")
        self.attn.save(path3)
    
    def load(self, path_prefix):
        new_self = self
        path1 = os.path.join(path_prefix, f"mlpl_1")
        mlpl1 =  self.mlpl1.load(path1)
        where = lambda x: x.mlpl1
        new_self = eqx.tree_at(where, new_self, mlpl1)
        
        path2 = os.path.join(path_prefix, f"mlpl_2")
        mlpl2 =  self.mlpl21.load(path2)
        where = lambda x: x.mlpl2
        new_self = eqx.tree_at(where, new_self, mlpl2)
        
        path3 = os.path.join(path_prefix, f"attn")
        attn =  self.attn.load(path3)
        where = lambda x: x.attn
        new_self = eqx.tree_at(where, new_self, attn)

        return new_self

        




