import pytest
import jax
import jax.numpy as jnp

import catfish.lvd.models.dist_layers as dl
import catfish.lvd.models.dist_utils as du

def test_shrd_mh_attention(dist_manager, prng_key):
    n_head, d_model, d_qk, d_v = 8, 128, 64, 64
    x = jax.random.normal(prng_key, (10, d_model))
    
    key1, key2 = jax.random.split(prng_key)
    attention_layer = dl.ShrdMHAttention(dist_manager, key1, d_model, n_head, d_qk, d_v)
    y1 = attention_layer(x)
    attention_layer.save("/test/shrdmhattention")
    
    attention_layer = dl.ShrdMHAttention(dist_manager, key2, d_model, n_head, d_qk, d_v)
    attention_layer = attention_layer.load("/test/shrdmhattention")
    y2 = attention_layer(x)
    
    assert jnp.allclose(y1, y2), "Attention outputs do not match after reload."

def test_shrd_conv(dist_manager, prng_key):
    x = jax.random.normal(prng_key, (8, 10, 10))  # Example input for convolution
    key1, key2 = jax.random.split(prng_key)

    conv_layer = dl.ShrdConv(dist_manager, key1, 3, 3, 8, 12)
    y1 = conv_layer(x)
    conv_layer.save("/test/shrdconv")
    
    conv_layer = dl.ShrdConv(dist_manager, key2, 3, 3, 8, 12)
    conv_layer = conv_layer.load("/test/shrdconv")
    y2 = conv_layer(x)
    
    assert jnp.allclose(y1, y2), "Convolution outputs do not match after reload."

def test_conv_res_block(dist_manager, prng_key):
    x = jax.random.normal(prng_key, (8, 10, 10))  # Example input
    key1, key2 = jax.random.split(prng_key)

    res_block = dl.ConvResBlock(dist_manager, key1, 8, 12)
    y1 = res_block(x)
    res_block.save("/test/convresblock")
    
    res_block = dl.ConvResBlock(dist_manager, key2, 8, 12)
    res_block = res_block.load("/test/convresblock")
    y2 = res_block(x)
    
    assert jnp.allclose(y1, y2), "Residual block outputs do not match after reload."

def test_transformer_block(dist_manager, prng_key):
    x = jax.random.normal(prng_key, (10, 128))  # Example input for transformer block
    key1, key2 = jax.random.split(prng_key)

    transformer = dl.TransformerBlock(dist_manager, key1, 128, 64, 32, 32, 8)
    y1 = transformer(x)
    transformer.save("/test/transformerblock")
    
    transformer = dl.TransformerBlock(dist_manager, key2, 128, 64, 32, 32, 8)
    transformer = transformer.load("/test/transformerblock")
    y2 = transformer(x)
    
    assert jnp.allclose(y1, y2), "Transformer block outputs do not match after reload."

def test_shrd_linear(dist_manager, prng_key):
    in_dim, out_dim = 64, 64
    x = jax.random.normal(prng_key, (in_dim,))

    key1, key2 = jax.random.split(prng_key)

    linear_layer = dl.ShrdLinear(dist_manager, key1, in_dim, out_dim)
    y1 = linear_layer(x)
    linear_layer.save("/test/shrdlinear")

    linear_layer = dl.ShrdLinear(dist_manager, key2, in_dim, out_dim)
    linear_layer = linear_layer.load("/test/shrdlinear")
    y2 = linear_layer(x)

    print(y1, y2)

    assert jnp.allclose(y1, y2), "Linear layer outputs do not match after reload."

