import pytest
import jax
import fs.memoryfs

import catfish.lvd.models.dist_utils as du

@pytest.fixture
def dist_manager():
    # Create a memory filesystem instance
    memory_fs = fs.memoryfs.MemoryFS()
    # Initialize your DistManager with the memory filesystem
    return du.DistManager(mesh_shape=(4, 2, 1), filesystem=memory_fs)

@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)

@pytest.fixture
def input_image(prng_key):
    # Create a dummy input image of shape [3 x 512 x 256]
    return jax.random.normal(prng_key, (3, 512, 256))