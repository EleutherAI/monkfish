import pytest
import jax
import jax.numpy as jnp
import jax.sharding as shrd
import equinox as eqx
import fs.memoryfs
import monkfish.lvd.models.dist_utils as du

@pytest.fixture
def dist_manager():
    mesh_shape = (8, 1, 1)  # Adjust based on your testing environment
    filesystem = fs.memoryfs.MemoryFS()
    return du.DistManager(mesh_shape, filesystem)

def test_init(dist_manager):
    assert isinstance(dist_manager, du.DistManager)
    assert dist_manager.pid == jax.process_index()
    assert dist_manager.nodes == jax.process_count()
    assert dist_manager.mesh_shape == (8, 1, 1)

def test_get_key(dist_manager):
    key = dist_manager.get_key(42)
    
    # Check type and shape
    assert isinstance(key, jax.Array)
    assert key.shape == (2,)
    assert key.dtype == jnp.uint32
    
    # Check sharding
    expected_sharding = dist_manager.uniform_sharding
    actual_sharding = key.sharding
    
    assert isinstance(actual_sharding, shrd.NamedSharding)
    assert actual_sharding.mesh == expected_sharding.mesh
    assert actual_sharding.spec == expected_sharding.spec

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


def test_save_load_pytree_2(dist_manager, tmp_path):
    file_name = str(tmp_path / "test_pytree.pkl")
    
    # Create a pytree with various data types
    pytree = {
        "float_array": jnp.array([1., 2., 3.]),
        "int_array": jnp.array([1, 2, 3], dtype=jnp.int32),
        "bool_array": jnp.array([True, False, True]),
        "nested": {
            "float": jnp.array([4., 5., 6.]),
            "int": jnp.array([4, 5, 6], dtype=jnp.int32)
        },
        "scalar_float": jnp.array(3.14),
        "scalar_int": jnp.array(42, dtype=jnp.int32),
        "scalar_bool": jnp.array(True)
    }
    
    # Create corresponding sharding pytree
    sharding_pytree = {
        "float_array": dist_manager.uniform_sharding,
        "int_array": dist_manager.uniform_sharding,
        "bool_array": dist_manager.uniform_sharding,
        "nested": {
            "float": dist_manager.uniform_sharding,
            "int": dist_manager.uniform_sharding
        },
        "scalar_float": dist_manager.uniform_sharding,
        "scalar_int": dist_manager.uniform_sharding,
        "scalar_bool": dist_manager.uniform_sharding
    }
    
    # Save the pytree
    dist_manager.save_pytree(pytree, sharding_pytree, file_name)
    
    # Load the pytree
    loaded_pytree = dist_manager.load_pytree(sharding_pytree, file_name)
    
    # Verify the structure and contents
    assert isinstance(loaded_pytree, dict)
    assert set(loaded_pytree.keys()) == set(pytree.keys())
    
    # Check each element
    assert jnp.allclose(pytree["float_array"], loaded_pytree["float_array"])
    assert jnp.all(pytree["int_array"] == loaded_pytree["int_array"])
    assert jnp.all(pytree["bool_array"] == loaded_pytree["bool_array"])
    
    assert jnp.allclose(pytree["nested"]["float"], loaded_pytree["nested"]["float"])
    assert jnp.all(pytree["nested"]["int"] == loaded_pytree["nested"]["int"])
    
    assert jnp.isclose(pytree["scalar_float"], loaded_pytree["scalar_float"])
    assert pytree["scalar_int"] == loaded_pytree["scalar_int"]
    assert pytree["scalar_bool"] == loaded_pytree["scalar_bool"]
    
    # Check dtypes
    assert loaded_pytree["float_array"].dtype == jnp.float32
    assert loaded_pytree["int_array"].dtype == jnp.int32
    assert loaded_pytree["bool_array"].dtype == jnp.bool_
    assert loaded_pytree["nested"]["float"].dtype == jnp.float32
    assert loaded_pytree["nested"]["int"].dtype == jnp.int32
    assert loaded_pytree["scalar_float"].dtype == jnp.float32
    assert loaded_pytree["scalar_int"].dtype == jnp.int32
    assert loaded_pytree["scalar_bool"].dtype == jnp.bool_


def test_get_pytree_sharding(dist_manager):
    # Create a simple pytree with mixed types
    pytree = {
        'a': jnp.array([1., 2., 3.]),
        'b': jnp.zeros((8,2),dtype=jnp.float32),
        'c': 'not an array',
        'd': {
            'e': jnp.array([5., 6.])
        }
    }

    # Apply different shardings to the arrays
    pytree['a'] = jax.device_put(pytree['a'], dist_manager.uniform_sharding)
    pytree['b'] = jax.device_put(pytree['b'], dist_manager.sharding(shrd.PartitionSpec('dp', None)))
    pytree['d']['e'] = jax.device_put(pytree['d']['e'], dist_manager.sharding(shrd.PartitionSpec('mp')))

    sharding_pytree = dist_manager.get_pytree_sharding(pytree)

    assert isinstance(sharding_pytree['a'], shrd.NamedSharding)
    assert sharding_pytree['a'].spec == shrd.PartitionSpec()
    
    assert isinstance(sharding_pytree['b'], shrd.NamedSharding)
    assert sharding_pytree['b'].spec == shrd.PartitionSpec('dp', None)
    
    assert sharding_pytree['c'] is None
    
    assert isinstance(sharding_pytree['d']['e'], shrd.NamedSharding)
    assert sharding_pytree['d']['e'].spec == shrd.PartitionSpec('mp')

def test_get_pytree_sharding_spec(dist_manager):
    # Create a simple pytree with mixed types
    pytree = {
        'a': jnp.array([1., 2., 3.]),
        'b': jnp.zeros((8,2),dtype=jnp.float32),
        'c': 'not an array',
        'd': {
            'e': jnp.array([5., 6.])
        }
    }

    # Apply different shardings to the arrays
    pytree['a'] = jax.device_put(pytree['a'], dist_manager.uniform_sharding)
    pytree['b'] = jax.device_put(pytree['b'], dist_manager.sharding(shrd.PartitionSpec('dp', None)))
    pytree['d']['e'] = jax.device_put(pytree['d']['e'], dist_manager.sharding(shrd.PartitionSpec('mp')))

    sharding_spec_pytree = dist_manager.get_pytree_sharding_spec(pytree)

    assert isinstance(sharding_spec_pytree['a'], shrd.PartitionSpec)
    assert sharding_spec_pytree['a'] == shrd.PartitionSpec()
    
    assert isinstance(sharding_spec_pytree['b'], shrd.PartitionSpec)
    assert sharding_spec_pytree['b'] == shrd.PartitionSpec('dp', None)
    
    assert sharding_spec_pytree['c'] is None
    
    assert isinstance(sharding_spec_pytree['d']['e'], shrd.PartitionSpec)
    assert sharding_spec_pytree['d']['e'] == shrd.PartitionSpec('mp')