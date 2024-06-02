import pytest
import jax

import catfish.lvd.models.dist_utils as du

@pytest.fixture
def dist_manager():
    return du.DistManager((4, 2, 1), "service-account-key.json", "lvd_test")

@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)

@pytest.fixture
def input_image(prng_key):
    # Create a dummy input image of shape [3 x 512 x 256]
    return jax.random.normal(prng_key, (3, 512, 256))