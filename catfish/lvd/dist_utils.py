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
    def __init__(self, key, mesh_shape):
        self.pid = jax.process_index()
        self.cpu_device = jax.local_devices("cpu")[0]
        self.local_accelerators = jax.local_devices()
        
        self.mesh_shape = mesh_shape
        self.physical_mesh = mesh_utils.create_device_mesh(
            mesh_shape, allow_split_physical_axes= True)
        
        self.mesh = shrd.Mesh(self.physical_mesh, ("dp","mp","fsdp"))
        
        with jax.default_device(self.cpu_device):
            host_key = jax.random.PRNGKey(key)
        
        uniform_sharding = shrd.NamedSharding(self.mesh, shrd.PartionSpec())

        self.prng_key = self.scatter(uniform_sharding)(host_key)
    
    def scatter(self, sharding):
        f = jax.jit(id, in_shardings=None, out_shardings=sharding)
        return f
    
    def gather(self, sharding):
        f = lambda x: jax.device_get(jax.jit(id, in_shardings=sharding, out_shardings=sharding))(x)
        return f

    def init_random_array(self, shape, std, sharding, key):
        cpu_array = self._init_array_cpu(key, std, shape)
        array = self.scatter(sharding)(cpu_array)
        return array

    def _init_array_cpu(self, key, std, shape):
        @jax.jit(device=self.cpu_device)
        def f(key, std, shape):
            return jax.random.normal(key, shape)*std
        cpu_f = jax.jit(f, device=self.cpu_device)
        cpu_array = cpu_f(key, std, shape)
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

"""
class ShardedLinear(eqx.Module):
    def __init__(self, key, host_id, sharding, shape, init_std, bias=False):
        self.host_id = host_id
        self.sharding = sharding
        self.shape = shape
        self.init_std = init_std

    def _init(self, key):
        #init array on CPU
        if host_id == 0:
            #We are the host
            array = jnp.random.normal(self.array)
        if not mhu.assert_equal(key):
            Exception("Different Keys Provided to different hosts at init")

        mesh = self.mesh()
        global_key_p_spec = shrd.PartitionSpec()
        self.sharded_key = mhu.host_local_array_to_global_array(key, mesh, global_key_p_spec)

        weight_p_spec = shrd.PartitionSpec("mp")
        bias_p_spec = shrd.PartitionSpec()
        
        @jax.jit
        def init_fn(shape,)

            self.sharded_key = mhu.host_local_array_to_global_array()
        self.weights, self.bias = 

    def _mesh(self):
        dp = self.sharding["dp"]
        mp = self.sharding["mp"]
        if (dp*mp == len(jax.devices)):
            raise Exception("Invalid sharding for device count")
        
        mesh_ids = mesh_utils.create_device_mesh((dp,mp))
        mesh = jax.sharding.Mesh(mesh_ids, ("mp"))
        return mesh
    
    def _weight_spec(self):
        return shrd.PartitionSpec("mp")

    def _bias_spec(self):
        return shrd.PartitionSpec()

    

    def load(param_dir):
        #Load specific partition
        shard_spec_path = os.path.join(param_dir, "shard_spec.pkl")
        with open(shard_spec_path,"rb") as f: 
            self.sharding = pikle.load(f)

        weight_path = os.path.join(param_dir, f"shard{self._shard_id()}weight.pkl")
        with open(weight_path,"rb") as f: 
            local_weights = pickle.load(
        #Merge partitions into single array
        self.weights = mhu.host_local_array_to_global_array()

        #Do same for bias

        #TODO:


    def save(param_dir)
        #NOTE: Saved weights might be replicated


class ShardedConv(eqx.Module):
"""


