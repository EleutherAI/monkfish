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

import numpy as np

import jax.numpy as jnp
import jax.tree_util as jtu

class DistManager:
    def __init__(self, mesh_shape, filesystem):
        self.pid = jax.process_index()
        self.nodes = jax.process_count()
        self.cpu_device = jax.local_devices(backend="cpu")[0]
        self.local_devices = jax.local_devices()
        self.local_mesh = shrd.Mesh(self.local_devices, ("local_devices",))
        self.local_sharding = shrd.NamedSharding(self.local_mesh, shrd.PartitionSpec(("local_devices",)))

        self.mesh_shape = mesh_shape
        self.physical_mesh = mesh_utils.create_device_mesh(
            mesh_shape, allow_split_physical_axes=True)

        self.mesh = shrd.Mesh(self.physical_mesh, ("dp", "mp", "fsdp"))

        self.uniform_sharding = shrd.NamedSharding(self.mesh, shrd.PartitionSpec())

        self.fs = filesystem
    
    def get_key(self, seed):
        uniform_sharding = shrd.NamedSharding(self.mesh, shrd.PartitionSpec())
        """
        with jax.default_device(self.cpu_device):
            host_key = jax.random.PRNGKey(key)
        prng_key = self.scatter(uniform_sharding)(host_key)
        """
        f = lambda: jax.random.PRNGKey(seed)
        prng_f = jax.jit(f, out_shardings=uniform_sharding)
        prng_key = prng_f()
        return prng_key
    
    def np_local_to_jax_global_batch(self, local_batch):
        local_batch_dim = local_batch.shape[0]
        array_dim = local_batch.shape[1:]

        dp_sharding = shrd.NamedSharding(self.mesh, shrd.PartitionSpec(("dp",)))
        global_batch_dim = local_batch_dim * jax.device_count()
        global_shape = (global_batch_dim,) + array_dim

        global_batch = jax.make_array_from_process_local_data(
            dp_sharding, local_batch, global_shape)

        assert global_batch.shape == global_shape
        assert global_batch.sharding == dp_sharding

        return global_batch
    
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
        flat_pytree, _ = jtu.tree_flatten(pytree)
        flat_sharding_pytree, _ = jtu.tree_flatten(sharding_pytree)

        gathered_leaves = [
            self.gather(sharding, leaf.dtype)(leaf) if leaf is not None else None
            for leaf, sharding in zip(flat_pytree, flat_sharding_pytree)
        ]

        if self.pid == 0:
            dir_name = os.path.dirname(file_name)
            if dir_name and not self.fs.exists(dir_name):
                self.fs.makedirs(dir_name, recreate=True)

            with self.fs.openbin(file_name, 'w') as blob:
                blob.write(pkl.dumps(gathered_leaves))
            print(f"Uploaded {file_name} to {type(self.fs).__name__} at {file_name}")

        mhu.sync_global_devices("save_pytree_sync")

    def load_pytree(self, sharding_pytree, file_name):
        with self.fs.openbin(file_name, 'r') as blob:
            gathered_leaves = pkl.loads(blob.read())

        flat_sharding_pytree, tree_def = jtu.tree_flatten(sharding_pytree)

        scattered_leaves = [
            self.scatter(sharding, leaf.dtype)(leaf) if leaf is not None else None
            for leaf, sharding in zip(gathered_leaves, flat_sharding_pytree)
        ]

        distributed_pytree = jtu.tree_unflatten(tree_def, scattered_leaves)

        mhu.sync_global_devices("load_pytree_sync")
        return distributed_pytree

    def get_pytree_sharding(self, pytree):
        def get_leaf_sharding(leaf):
            if isinstance(leaf, jax.Array):
                return leaf.sharding
            else:
                return None

        return jax.tree_util.tree_map(get_leaf_sharding, pytree)

    def get_pytree_sharding_spec(self, pytree):
        def get_leaf_partition_spec(leaf):
            if isinstance(leaf, jax.Array):
                return leaf.sharding.spec
            else:
                return None

        return jax.tree_util.tree_map(get_leaf_partition_spec, pytree)
    