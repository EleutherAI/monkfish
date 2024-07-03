import pytest
import jax
import jax.numpy as jnp
import jax.sharding as shrd
import equinox as eqx
import fs.memoryfs
from your_module import DistManager  # Replace 'your_module' with the actual module name
import catfish.lvd.diffusion_ae

@pytest.fixture
def dist_manager():
    mesh_shape = (8, 1, 1)  # Adjust based on your testing environment
    filesystem = fs.memoryfs.MemoryFS()
    return catfish.lvd.diffusion_ae.DistManager(mesh_shape, filesystem)

def test_init(dist_manager):
    assert isinstance(dist_manager, DistManager)
    assert dist_manager.pid == jax.process_index()
    assert dist_manager.nodes == jax.process_count()
    assert dist_manager.mesh_shape == (1, 1, 1)

def test_get_key(dist_manager):
    key = dist_manager.get_key(42)
    assert isinstance(key, jax.random.PRNGKey)

def test_sharding(dist_manager):
    partition_spec = shrd.PartitionSpec("dp", "mp")
    sharding = dist_manager.sharding(partition_spec)
    assert isinstance(sharding, shrd.NamedSharding)

def test_scatter_gather(dist_manager):
    sharding = dist_manager.uniform_sharding
    x = jnp.array([1., 2., 3.])
    scattered = dist_manager.scatter(sharding, jnp.float32)(x)
    gathered = dist_manager.gather(sharding, jnp.float32)(scattered)
    assert jnp.allclose(x, gathered)

def test_init_randn_array(dist_manager):
    shape = (3, 3)
    std = 1.0
    sharding = dist_manager.uniform_sharding
    key = jax.random.PRNGKey(0)
    array = dist_manager.init_randn_array(shape, std, sharding, key)
    assert array.shape == shape

def test_init_pytree_cpu(dist_manager):
    def closure():
        return {"a": jnp.array([1., 2., 3.]), "b": jnp.array([4., 5., 6.])}
    pytree = dist_manager.init_pytree_cpu(closure)
    assert isinstance(pytree, dict)
    assert "a" in pytree and "b" in pytree

@pytest.mark.parametrize("array", [
    jnp.array([1., 2., 3.]),
    None
])
def test_save_load_array(dist_manager, array, tmp_path):
    file_name = str(tmp_path / "test_array.pkl")
    sharding = dist_manager.uniform_sharding
    
    dist_manager.save_array(array, sharding, file_name)
    loaded_array = dist_manager.load_array(sharding, file_name)
    
    if array is not None:
        assert jnp.allclose(array, loaded_array)
    else:
        assert loaded_array is None

def test_save_load_pytree(dist_manager, tmp_path):
    file_name = str(tmp_path / "test_pytree.pkl")
    pytree = {"a": jnp.array([1., 2., 3.]), "b": jnp.array([4., 5., 6.])}
    sharding_pytree = {"a": dist_manager.uniform_sharding, "b": dist_manager.uniform_sharding}
    
    dist_manager.save_pytree(pytree, sharding_pytree, file_name)
    loaded_pytree = dist_manager.load_pytree(sharding_pytree, file_name)
    
    assert isinstance(loaded_pytree, dict)
    assert "a" in loaded_pytree and "b" in loaded_pytree
    assert jnp.allclose(pytree["a"], loaded_pytree["a"])
    assert jnp.allclose(pytree["b"], loaded_pytree["b"])
