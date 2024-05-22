import functools
import os

import pickle as pkl

import equinox as eqx
import jax
import jax.experimental.multihost_utils as mhu
import jax.experimental.mesh_utils as mesh_utils
import jax.experimental.shard_map as shard_map
import jax.sharding as shrd



class DistManager:
    def __init__(self, mesh_shape):
        self.pid = jax.process_index()
        self.cpu_device = jax.local_devices("cpu")[0]
        self.local_accelerators = jax.local_devices()
        
        self.mesh_shape = mesh_shape
        self.physical_mesh = mesh_utils.create_device_mesh(
            mesh_shape, allow_split_physical_axes= True)
        
        self.mesh = shrd.Mesh(self.physical_mesh, ("dp","mp","fsdp"))
        
        self.uniform_sharding = shrd.NamedSharding(self.mesh, shrd.PartionSpec())
    
    def get_key(self, key):
        uniform_sharding = shrd.NamedSharding(self.mesh, shrd.PartionSpec())
        with jax.default_device(self.cpu_device):
            host_key = jax.random.PRNGKey(key)
        prng_key = self.scatter(uniform_sharding)(host_key)
        return prng_key
    
    def sharding(self, partition_spec):
        return shrd.NamedSharding(self.mesh, partition_spec)
    
    def scatter(self, sharding):
        f = jax.jit(id, in_shardings=None, out_shardings=sharding)
        return f
    
    def gather(self, sharding):
        f = lambda x: jax.device_get(jax.jit(id, in_shardings=sharding, out_shardings=None))(x)
        return f

    def init_randn_array(self, shape, std, sharding, key):
        cpu_array = self._init_randn_cpu(key, std, shape)
        array = self.scatter(sharding)(cpu_array)
        return array

    def init_pytree_cpu(self, closure):
        f = jax.jit(closure, device=self.cpu_device)
        return f()

    def _init_randn_cpu(self, key, std, shape):
        @jax.jit(device=self.cpu_device)
        def f(key, std, shape):
            return jax.random.normal(key, shape)*std
        cpu_array = f(key, std, shape)
        return cpu_array
        
    def save_array(self, array, sharding, path):
        local_array = self.gather(sharding)(array)

        #Only have first process actually write to disk
        if self.pid != 0:
            path = "/dev/null"
        
        with open(path, "wb") as f:
            pkl.dump(f ,local_array)
        
        mhu.sync_global_devices("save_sync")
    
    def load_array(self, sharding, path):
        with jax.default_device(self.cpu_device):
            with open(path, "rb") as f:
                local_array = pkl.load(f)
            array = self.scatter(sharding)(local_array)
        mhu.sync_global_devices("load_sync")
        return array
