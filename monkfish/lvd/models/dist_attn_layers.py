import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx

import monkfish.lvd.models.dist_utils as du
import monkfish.lvd.models.dist_layers as dl

def mha(x, y, q, k, v, o, mask):
    par_sha = jax.vmap(sha, in_axes=(None, None, 0, 0, 0, None))

    z = par_sha(x, y, q, k, v, mask)
    output = jnp.einsum("ijk,ikl->jl", z, o)
    return output

def sha(x, y, q, k, v, mask):
    pre_qs = jnp.einsum("ij,jk->ik", x, q)
    rot_qs = rope_embed(pre_qs, q.shape[1])
    
    pre_ks = jnp.einsum("ij,jk->ik", y, k)
    rot_ks = rope_embed(pre_ks, k.shape[1])
    
    vs = jnp.einsum("ij,jk->ik", y, v)

    unmasked_attention = jnp.einsum("ik,jk->ij", rot_qs, rot_ks)
    masked_attention = unmasked_attention + mask
    attention_weights = jax.nn.softmax(masked_attention)

    output = jnp.einsum("ij,jk->ik", attention_weights, vs)
    
    return output

def rope_embed(x, d_qk, theta_factor, qk_layer_norm=None):
    d_pos = x.shape[0]
    d_rope = x.shape[1] // 2
    rate_vector = theta_factor * (-jnp.arange(0, d_rope) / d_rope)
    pos_vector = jnp.arange(0, d_pos)

    rot_factor = jnp.einsum("i,j->ij", pos_vector, rate_vector)
    sin_factor, cos_factor = jnp.sin(rot_factor), jnp.cos(rot_factor)

    x1, x2 = x[:, :d_rope], x[:, d_rope:]

    y1 = cos_factor * x1 - sin_factor * x2
    y2 = sin_factor * x1 + cos_factor * x2

    y = jnp.concatenate([y1, y2], axis=1)
    
    if qk_layer_norm is None:
        y = y / (d_qk ** (1/4))
    else:
        raise NotImplementedError
    
    return y

def causal_mask(size):
    mask = jnp.tril(jnp.ones((size, size)))
    mask = mask - 1
    mask = mask * 1000000000
    return mask

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
        return dl.make_f_dict(pre_dict, self.dist_manager)
    
    def __init__(self, dist_manager, key, d_model, 
                 n_head, d_qk, d_v, qk_layer_norm=False, theta_factor=10000):
        keys = jax.random.split(key, 6)
    
        self.scale = 1/jnp.sqrt(n_head*d_v*d_model)

        self.dist_manager = dist_manager

        self.q = self.dist_manager.init_randn_array(
            (n_head, d_model, d_qk), 1, self._f_dict()["q"]["sharding"], keys[0])
        
        self.k = self.dist_manager.init_randn_array(
            (n_head, d_model, d_qk), 1, self._f_dict()["k"]["sharding"], keys[1])
        
        self.v = self.dist_manager.init_randn_array(
            (n_head, d_model, d_v), 1, self._f_dict()["v"]["sharding"], keys[2])
        
        self.o = self.dist_manager.init_randn_array(
            (n_head, d_v, d_model), 1, self._f_dict()["o"]["sharding"], keys[3])
        
        self.theta_factor = self.dist_manager.init_randn_array(
            (), 0, self._f_dict()["theta_factor"]["sharding"], keys[4])
        self.theta_factor += theta_factor
        
        if qk_layer_norm:
            self.qk_layer_norm = self.dist_manager.init_randn_array(
                (n_head,), 0, self._f_dict()["qk_layer_norm"]["sharding"], keys[5])
        else:
            self.qk_layer_norm = None

    def __call__(self, x, y=None):
        if y is None:
            y = x
        seq_length = x.shape[0]
        mask = causal_mask(seq_length)
        output = mha(x, y, self.q, self.k, self.v, self.o, mask) * self.scale
        return output

class TransformerBlock(eqx.Module):
    mlpl1: dl.ShrdLinear
    mlpl2: dl.ShrdLinear
    attn: ShrdMHAttention

    def __init__(self, dist_manager, key, res_dim, mlp_dim, 
                 qk_dim, v_dim, n_head):
        self.mlpl1 = dl.ShrdLinear(dist_manager, key, res_dim, mlp_dim)
        self.mlpl2 = dl.ShrdLinear(dist_manager, key, mlp_dim, res_dim)
        self.attn = ShrdMHAttention(dist_manager, key, res_dim, n_head, qk_dim, v_dim)
    
    def _norm(self, x):
        m = jnp.mean(x, axis=1)
        s = jnp.std(x, axis=1)
        y = (x-m[:, jnp.newaxis])/(s[:, jnp.newaxis])
        return y
    
    #[pos x res_dim] -> [pos x res_dim]
    def __call__(self, x, y=None):
        h1 = self._norm(x)
        h2 = jax.vmap(self.mlpl1)(h1)
        h3 = jax.vmap(self.mlpl2)(h2)
        h4 = self.attn(h1, y)
        output = (h3 + h4)
        return output
