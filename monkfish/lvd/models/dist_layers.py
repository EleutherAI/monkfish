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



