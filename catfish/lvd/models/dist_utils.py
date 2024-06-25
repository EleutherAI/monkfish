import functools
import os

import google.cloud.storage as gcs

import pickle as pkl

import equinox as eqx
import jax
import jax.experimental.multihost_utils as mhu
import jax.experimental.mesh_utils as mesh_utils
import jax.experimental.shard_map as shard_map
import jax.sharding as shrd

import jax.numpy as jnp

class DistManager:
    def __init__(self, mesh_shape, filesystem):
        self.pid = jax.process_index()
        self.cpu_device = jax.local_devices(backend="cpu")[0]
        self.local_accelerators = jax.local_devices()

        self.mesh_shape = mesh_shape
        self.physical_mesh = mesh_utils.create_device_mesh(
            mesh_shape, allow_split_physical_axes=True)

        self.mesh = shrd.Mesh(self.physical_mesh, ("dp", "mp", "fsdp"))

        self.uniform_sharding = shrd.NamedSharding(self.mesh, shrd.PartitionSpec())

        self.fs = filesystem
    
    def get_key(self, key):
        uniform_sharding = shrd.NamedSharding(self.mesh, shrd.PartitionSpec())
        """
        with jax.default_device(self.cpu_device):
            host_key = jax.random.PRNGKey(key)
        prng_key = self.scatter(uniform_sharding)(host_key)
        """
        f = lambda: jax.random.PRNGKey(key)
        prng_f = jax.jit(f, out_shardings=uniform_sharding)
        prng_key = prng_f()
        return prng_key
    
    def sharding(self, partition_spec):
        return shrd.NamedSharding(self.mesh, partition_spec)
    
    def scatter(self, sharding, dtype):
        g = lambda x: x.astype(dtype)
        f = jax.jit(g, in_shardings=None, out_shardings=sharding)
        return f
    
    def gather(self, sharding, dtype):
        g = lambda x: x.astype(dtype)
        f = lambda x: jax.device_get(jax.jit(g, in_shardings=sharding, out_shardings=None))(x)
        return f

    def init_randn_array(self, shape, std, sharding, key):
        cpu_array = self._init_randn_cpu(key, std, shape)
        array = self.scatter(sharding, jnp.float32)(cpu_array)
        return array

    def init_pytree_cpu(self, closure):
        f = jax.jit(closure, device=self.cpu_device)
        return f()

    def _init_randn_cpu(self, key, std, shape):
        with jax.default_device(self.cpu_device):
            @functools.partial(jax.jit, static_argnums=(2,))
            def f(key, std, shape):
                return jax.random.normal(key, shape)*std
            cpu_array = f(key, std, shape)
        return cpu_array

    def save_array(self, array, sharding, file_name):
        if array is not None:
            local_array = self.gather(sharding, jnp.float32)(array)
        else:
            local_array = None
        
        # Only have first process actually write to the filesystem
        if self.pid == 0:
            # Ensure directory exists before writing
            dir_name = os.path.dirname(file_name)
            if dir_name and not self.fs.exists(dir_name):
                # Create directories recursively if they do not exist
                self.fs.makedirs(dir_name, recreate=True)

            with self.fs.openbin(file_name, 'w') as blob:
                blob.write(pkl.dumps(local_array))
            print(f"Uploaded {file_name} to {type(self.fs).__name__} at {file_name}")
        mhu.sync_global_devices("save_sync")

    def load_array(self, sharding, file_name):
        with self.fs.openbin(file_name, 'r') as blob:
            local_array_pkl = blob.read()
        local_array = pkl.loads(local_array_pkl)
        
        if local_array is not None:
            array = self.scatter(sharding, jnp.float32)(local_array)
        else:
            array = None 
        mhu.sync_global_devices("load_sync")
        return array
    
    def save_pytree(self, pytree, sharding_pytree, file_name):
        pass
    
    def load_pytree(self, sharding_pytree, file_name):
        data_pytree = None
        return data_pytree