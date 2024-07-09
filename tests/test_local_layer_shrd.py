import pytest
import jax
import jax.numpy as jnp

import equinox as eqx

import monkfish.lvd.models.dist_layers as dl
import monkfish.lvd.models.dist_utils as du

def test_shrd_mh_attention_save_load(dist_manager, prng_key):
    n_head, d_model, d_qk, d_v = 8, 128, 64, 64
    x = jax.random.normal(prng_key, (10, d_model))
    
    key1, key2 = jax.random.split(prng_key)

    attention_layer = dl.ShrdMHAttention(dist_manager, key1, d_model, n_head, d_qk, d_v)
    sharding_pytree = dist_manager.get_pytree_sharding(attention_layer)
    dist_manager.save_pytree(attention_layer, sharding_pytree, "/test/shrdmhattention")
    
    y1 = attention_layer(x)
    
    new_attention_layer = dl.ShrdMHAttention(dist_manager, key2, d_model, n_head, d_qk, d_v)
    new_sharding_pytree = dist_manager.get_pytree_sharding(new_attention_layer)
    loaded_attention_layer = dist_manager.load_pytree(new_sharding_pytree, "/test/shrdmhattention")

    # Ensure the loaded layer has the same structure as the original
    assert eqx.tree_equal(attention_layer, loaded_attention_layer), "Loaded layer structure does not match original."

    # Compare outputs
    y2 = loaded_attention_layer(x)
    
    assert jnp.allclose(y1, y2), "Attention outputs do not match after reload."


def test_shrd_conv_save_load(dist_manager, prng_key):
    x = jax.random.normal(prng_key, (8, 10, 10))  # Example input for convolution
    key1, key2 = jax.random.split(prng_key)

    conv_layer = dl.ShrdConv(dist_manager, key1, 3, 3, 8, 12)
    sharding_pytree = dist_manager.get_pytree_sharding(conv_layer)
    y1 = conv_layer(x)
    dist_manager.save_pytree(conv_layer, sharding_pytree, "/test/shrdconv")
    
    new_conv_layer = dl.ShrdConv(dist_manager, key2, 3, 3, 8, 12)
    new_sharding_pytree = dist_manager.get_pytree_sharding(new_conv_layer)
    loaded_conv_layer = dist_manager.load_pytree(new_sharding_pytree, "/test/shrdconv")
    
    assert eqx.tree_equal(conv_layer, loaded_conv_layer), "Loaded conv layer structure does not match original."
    
    y2 = loaded_conv_layer(x)
    assert jnp.allclose(y1, y2), "Convolution outputs do not match after reload."

def test_conv_res_block_save_load(dist_manager, prng_key):
    x = jax.random.normal(prng_key, (8, 10, 10))  # Example input
    key1, key2 = jax.random.split(prng_key)

    res_block = dl.ConvResBlock(dist_manager, key1, 8, 12)
    sharding_pytree = dist_manager.get_pytree_sharding(res_block)
    y1 = res_block(x)
    dist_manager.save_pytree(res_block, sharding_pytree, "/test/convresblock")
    
    new_res_block = dl.ConvResBlock(dist_manager, key2, 8, 12)
    new_sharding_pytree = dist_manager.get_pytree_sharding(new_res_block)
    loaded_res_block = dist_manager.load_pytree(new_sharding_pytree, "/test/convresblock")
    
    assert eqx.tree_equal(res_block, loaded_res_block), "Loaded residual block structure does not match original."
    
    y2 = loaded_res_block(x)
    assert jnp.allclose(y1, y2), "Residual block outputs do not match after reload."

def test_transformer_block_save_load(dist_manager, prng_key):
    x = jax.random.normal(prng_key, (10, 128))  # Example input for transformer block
    key1, key2 = jax.random.split(prng_key)

    transformer = dl.TransformerBlock(dist_manager, key1, 128, 64, 32, 32, 8)
    sharding_pytree = dist_manager.get_pytree_sharding(transformer)
    y1 = transformer(x)
    dist_manager.save_pytree(transformer, sharding_pytree, "/test/transformerblock")
    
    new_transformer = dl.TransformerBlock(dist_manager, key2, 128, 64, 32, 32, 8)
    new_sharding_pytree = dist_manager.get_pytree_sharding(new_transformer)
    loaded_transformer = dist_manager.load_pytree(new_sharding_pytree, "/test/transformerblock")
    
    assert eqx.tree_equal(transformer, loaded_transformer), "Loaded transformer block structure does not match original."
    
    y2 = loaded_transformer(x)
    assert jnp.allclose(y1, y2), "Transformer block outputs do not match after reload."

def test_shrd_linear_save_load(dist_manager, prng_key):
    in_dim, out_dim = 64, 64

    prng_key = dist_manager.get_key(42)

    x = jax.random.normal(prng_key, (in_dim,))

    key1, key2 = jax.random.split(prng_key)

    linear_layer = dl.ShrdLinear(dist_manager, key1, in_dim, out_dim)
    sharding_pytree = dist_manager.get_pytree_sharding(linear_layer)
    y1 = linear_layer(x)
    dist_manager.save_pytree(linear_layer, sharding_pytree, "/test/shrdlinear")

    new_linear_layer = dl.ShrdLinear(dist_manager, key2, in_dim, out_dim)
    new_sharding_pytree = dist_manager.get_pytree_sharding(new_linear_layer)
    loaded_linear_layer = dist_manager.load_pytree(new_sharding_pytree, "/test/shrdlinear")
    print(sharding_pytree)
    print(new_sharding_pytree)
    
    assert eqx.tree_equal(linear_layer, loaded_linear_layer), "Loaded linear layer structure does not match original."
    
    y2 = loaded_linear_layer(x)
    assert jnp.allclose(y1, y2), "Linear layer outputs do not match after reload."
