import pytest
import jax
import jax.numpy as jnp

import monkfish.lvd.models.dist_layers as dl
import monkfish.lvd.models.dist_utils as du
import monkfish.lvd.models.dist_autoreg_diffusion as dad

def test_split_transformer_ardm_initialization(dist_manager, prng_key):
    split_transformer_ardm = dad.SplitTransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
    assert isinstance(split_transformer_ardm, dad.SplitTransformerARDM), "SplitTransformer initialization failed"

def test_split_transformer_ardm_forward_pass(dist_manager, prng_key):
    split_transformer_ardm = dad.SplitTransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
        
    txt = jnp.ones((128,), dtype=jnp.int32)
    noise_x = jax.random.normal(prng_key,(128, 128))
    true_x = jax.random.normal(prng_key,(128, 128))

    denoise = split_transformer_ardm((txt, true_x), noise_x, 0.0)
    assert denoise.shape == true_x.shape, f"Unexpected output shape from SplitTransformerARDM: {denoise.shape}"

def test_save_load_consistency_split(dist_manager, prng_key):
    key = jax.random.split(prng_key, 2)
    
    txt = jnp.ones((128,), dtype=jnp.int32)
    noise_x = jax.random.normal(prng_key,(128, 128))
    true_x = jax.random.normal(prng_key,(128, 128))
    
    split_transformer_ardm_initial = dad.SplitTransformerARDM(dist_manager, key[0], res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)

    denoise_initial = split_transformer_ardm_initial((txt, true_x), noise_x, 0.0)

    sharding_initial = dist_manager.get_pytree_sharding(split_transformer_ardm_initial)
    dist_manager.save_pytree(split_transformer_ardm_initial, sharding_initial, "/test/split_transformer_ardm")

    split_transformer_ardm_reloaded = dad.SplitTransformerARDM(dist_manager, key[1], res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)

    sharding_reloaded = dist_manager.get_pytree_sharding(split_transformer_ardm_reloaded)
    split_transformer_ardm_reloaded = dist_manager.load_pytree(sharding_reloaded, "/test/split_transformer_ardm")

    denoise_reloaded = split_transformer_ardm_reloaded((txt, true_x), noise_x, 0.0)

    assert jnp.allclose(denoise_initial, denoise_reloaded, atol=1e-5), "SplitTransformerARDM outputs do not match after reload."

def test_split_transformer_ardm_causality(dist_manager, prng_key):
    split_transformer_ardm = dad.SplitTransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
    
    key1, key2 = jax.random.split(prng_key)
    
    txt = jnp.ones((128,), dtype=jnp.int32)
    noise_x = jax.random.normal(key1, (128, 128))
    true_x = jax.random.normal(key2, (128, 128))
    
    original_output = split_transformer_ardm((txt, true_x), noise_x, 0.0)
    
    modified_true_x = true_x.at[-1].set(jnp.zeros_like(true_x[-1]))
    
    modified_output = split_transformer_ardm((txt, modified_true_x), noise_x, 0.0)
    
    assert jnp.allclose(original_output[:-1], modified_output[:-1]), "Non-causal behavior detected: earlier outputs changed"
    assert not jnp.allclose(original_output[-1], modified_output[-1]), "Last output should have changed"

def test_split_transformer_ardm_stability(dist_manager, prng_key):
    split_transformer_ardm = dad.SplitTransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=10, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
    
    key1, key2 = jax.random.split(prng_key)
    
    txt = jnp.ones((128,), dtype=jnp.int32)
    noise_x = jax.random.normal(key1, (128, 128))
    true_x = jax.random.normal(key2, (128, 128))
    
    for _ in range(3):
        output = split_transformer_ardm((txt, true_x), noise_x, 0.0)
        
        assert not jnp.isnan(output).any(), "NaN values detected in the output"
        assert not jnp.isinf(output).any(), "Inf values detected in the output"
        
        assert jnp.max(jnp.abs(output)) < 100, "Output magnitude is too large"
        
        true_x = output

def test_split_transformer_ardm_generate_mode(dist_manager, prng_key):
    split_transformer_ardm = dad.SplitTransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
    
    txt = jnp.ones((128,), dtype=jnp.int32)
    true_x = jax.random.normal(prng_key, (127, 128))  # One less than in train mode
    noise_x = jax.random.normal(prng_key, (128, 128))  # One more than true_x
    
    output = split_transformer_ardm((txt, true_x), noise_x, 0.0, mode="generate")
    assert output.shape == (128,), f"Unexpected output shape in generate mode: {output.shape}"
