import equinox as eqx
import dist_utils as du

class ShrdMHAttention(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    q: jax.Array
    k: jax.Array
    v: jax.Array
    
    def __init__(self, dist_manager, key):
        self.dist_manager = dist_manager
    
    def shard(self, sharding_spec):
    
    def __call__(self, x):

class ShrdConv(eqx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    kernel: jax.Array
    bias: jax.Array
    
    def __init__(self, dist_manager, key):
        self.dist_manager = dist_manager
    
    def shard(self, sharding_spec):
    
    def __call__(self, x):

class ShrdLinear(equx.Module):
    dist_manager: du.DistManager = eqx.field(static=True)
    weight: jax.Array
    bias: jax.Array
    
    def __init__(self, dist_manager, key):
        self.dist_manager = dist_manager

    def shard(sharding_spec)
    
    def __call__(self, x):
        jnp.einsum(x






