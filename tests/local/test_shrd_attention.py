import pytest
import jax
import jax.numpy as jnp
import numpy as np

import monkfish.lvd.models.dist_layers as dl
import monkfish.lvd.models.dist_utils as du

@pytest.fixture
def attention_layer(dist_manager):
    n_head, d_model, d_qk, d_v = 8, 128, 64, 64
    key = jax.random.PRNGKey(0)
    return dl.ShrdMHAttention(dist_manager, key, d_model, n_head, d_qk, d_v)

def test_output_magnitude(dist_manager, attention_layer):
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (10, 128))  # 10 tokens, 128 dimensions
    
    y = attention_layer(x)
    
    # Check that output has the same shape as input
    assert x.shape == y.shape
    
    # Check that the magnitude of the output is similar to the input
    input_magnitude = jnp.linalg.norm(x)
    output_magnitude = jnp.linalg.norm(y)
    
    # The output magnitude should be within an order of magnitude of the input
    assert 0.1 * input_magnitude < output_magnitude < 10 * input_magnitude

def test_causality(dist_manager, attention_layer):
    key = jax.random.PRNGKey(2)
    seq_length = 20
    d_model = 128
    x = jax.random.normal(key, (seq_length, d_model))
    
    y = attention_layer(x)
    
    # Create a distinctive signal at a specific position
    signal_position = 10
    new_x = x.at[signal_position].set(jnp.ones(d_model) * 10)
    
    new_y = attention_layer(new_x)
    
    # Check that the signal only affects subsequent positions
    for i in range(seq_length):
        if i < signal_position:
            # Positions before the signal should not be affected
            assert jnp.allclose(y[i], new_y[i], atol=1e-5)
        else:
            # Positions after (and including) the signal can be affected
            assert not jnp.allclose(y[i], new_y[i], atol=1e-5)
