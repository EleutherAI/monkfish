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
    x = jax.random.normal(key, (seq_length, 128))
    
    y = attention_layer(x)
    
    # Function to compute attention and extract weights
    def get_attention_weights(x):
        # This is a simplified version and might need adjustment based on your actual implementation
        q = jnp.einsum("ij,hjk->hik", x, attention_layer.q)
        k = jnp.einsum("ij,hjk->hik", x, attention_layer.k)
        attention = jnp.einsum("hik,hjk->hij", q, k)
        return jax.nn.softmax(attention, axis=-1)
    
    weights = get_attention_weights(x)
    
    # Check that weights form a lower triangular matrix for each head
    for head_weights in weights:
        triu_sum = jnp.sum(jnp.triu(head_weights, k=1))
        assert jnp.allclose(triu_sum, 0, atol=1e-6), "Upper triangle of attention weights is not zero"

def test_attention_uniformity(dist_manager, attention_layer):
    key = jax.random.PRNGKey(3)
    x = jax.random.normal(key, (10, 128))
    
    # Function to compute attention weights
    def get_attention_weights(x):
        q = jnp.einsum("ij,hjk->hik", x, attention_layer.q)
        k = jnp.einsum("ij,hjk->hik", x, attention_layer.k)
        attention = jnp.einsum("hik,hjk->hij", q, k)
        return jax.nn.softmax(attention, axis=-1)
    
    weights = get_attention_weights(x)
    
    # Check that weights for each token sum to 1
    weight_sums = jnp.sum(weights, axis=-1)
    assert jnp.allclose(weight_sums, 1.0, atol=1e-6), "Attention weights do not sum to 1 for each token"

def test_attention_head_diversity(dist_manager, attention_layer):
    key = jax.random.PRNGKey(4)
    x = jax.random.normal(key, (10, 128))
    
    def get_attention_weights(x):
        q = jnp.einsum("ij,hjk->hik", x, attention_layer.q)
        k = jnp.einsum("ij,hjk->hik", x, attention_layer.k)
        attention = jnp.einsum("hik,hjk->hij", q, k)
        return jax.nn.softmax(attention, axis=-1)
    
    weights = get_attention_weights(x)
    
    # Compute pairwise correlations between attention heads
    correlations = []
    n_heads = weights.shape[0]
    for i in range(n_heads):
        for j in range(i+1, n_heads):
            corr = jnp.corrcoef(weights[i].ravel(), weights[j].ravel())[0, 1]
            correlations.append(corr)
    
    # Check that not all heads are perfectly correlated
    assert not jnp.allclose(correlations, 1.0, atol=1e-2), "All attention heads are too similar"
