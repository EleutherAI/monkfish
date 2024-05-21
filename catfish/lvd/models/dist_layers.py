import os

import jax
import jax.numpy as jnp
import jax.sharding as shrd
import jax.lax as lax
import equinox as eqx

import dist_utils as du

#TODO: Subclass eqx.Module properly

def make_f_dict(pre_dict, dist_manager):
    f_dict = {}
    for name, (p_spec, path) in pre_dict.items():
        partition_spec = shrd.PartitionSpec(*p_spec)
        sharding = dist_manager.sharding(partition_spec)
        f_dict[name] = {
            "sharding": sharding,
            "path_fn": lambda p: os.path.join(p, path)
        }
    return f_dict

class ShrdMHAttention(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    q: jax.Array
    k: jax.Array
    v: jax.Array
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
    
    def __init__(self, dist_manager, d, key):
        self.dist_manager = dist_manager
        
        #Init q_k)v 
        scale = scale
        shape = (h, w, in_dim, out_dim)
        std = 1
        weight_pspec = shrd.PartitionSpec(("mp","fsdp"), None)
        kernel_shrd = self.dist_manager.sharding(weight_pspec)
        self.kernel = self.dist_manager.init_randn_array(self, shape, std, kernel_shrd)
    
    def __call__(self, x):
        pass

#TODO: Support dilation
class ShrdConv(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    kernel: jax.Array
    bias: jax.Array
    scale: jax.Array = eqx.field(static=True)
    padding: str = eqx.field(static=True)
    
    def __init__(self, dist_manager, h, w, in_dim, out_dim, 
                 padding="SAME", bias=False):
        self.dist_manager = dist_manager
        
        self.scale = 1/jax.numpy.sqrt(h*w*in_dim)

        self.padding = padding
        
        #Init kernel
        scale = scale
        shape = (h, w, in_dim, out_dim)
        std = 1
        kernel_pspec = shrd.PartitionSpec(("mp","fsdp"), None)
        kernel_shrd = self.dist_manager.sharding(kernel_pspec)
        self.kernel = self.dist_manager.init_randn_array(self, shape, std, kernel_shrd)
        
        #Init bias
        if bias:
            shape = (out_dim,)
            std = 0
            bias_pspec = shrd.PartitionSpec(None)
            bias_shrd = self.dist_manager.sharding(bias_pspec)
            self.bias = self.dist_manager.init_randn_array(self, shape, std, bias_shrd)
        else:
            self.bias = None
    
    def __call__(self, x):
        y = lax.conv_with_general_padding(x, self.ei)
        if self.bias is not None:
            y = y + self.bias
        return y
    
    def save(self, path_prefix):
        for key, value in self._f_dict():
            array = getattr(self, key)
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"](path_prefix)
            self.dist_manager.save_array(
                array, sharding, path)
    
    def load(self, path_prefix):
        for key, value in self._f_dict():
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"](path)
            array = self.dist_manager.load_array(
                sharding, path)
            setattr(self, key, array)

class ShrdLinear(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    weight: jax.Array
    bias: jax.Array | None
    scale: jax.Array = eqx.field(static=True)
    
    def _f_dict(self):
        pre_dict = {
            "weight": ((("mp","fsdp"), None), "weight.pkl"),
            "bias": ((None), "bias.pkl"),
            "scale":((None), "scale.pkl")

        }
        return make_f_dict(pre_dict, self.dist_manager)
    
    def __init__(self, dist_manager, in_dim, out_dim, bias=True):
        self.dist_manager = dist_manager

        self.scale = 1/jnp.sqrt(out_dim)

        #Init weight
        shape = (in_dim, out_dim)
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
    
    def __call__(self, x):
        y = jnp.einsum("i,ij->j",x, self.weight)*self.scale
        if self.bias is not None:
            y = y + self.bias
        return y
    
    def save(self, path_prefix):
        for key, value in self._f_dict():
            array = getattr(self, key)
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"](path_prefix)
            self.dist_manager.save_array(
                array, sharding, path)
    
    def load(self, path_prefix):
        for key, value in self._f_dict():
            path = value["path_fn"](path_prefix)
            sharding = value["sharding"](path)
            array = self.dist_manager.load_array(
                sharding, path)
            setattr(self, key, array)






