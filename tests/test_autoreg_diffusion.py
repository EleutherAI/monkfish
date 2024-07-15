import pytest
import jax
import jax.numpy as jnp

import monkfish.lvd.models.dist_layers as dl
import monkfish.lvd.models.dist_utils as du
import monkfish.lvd.models.dist_autoreg_diffusion as dad

def test_transformer_ardm_initialization(dist_manager, prng_key):
    transformer_ardm = dad.TransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
    assert isinstance(transformer_ardm, dad.TransformerARDM), "Transformer initialization failed"

def test_transformer_ardm_forward_pass(dist_manager, prng_key, input_image):
    transformer_ardm = dad.TransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
        
    txt = jnp.ones((128,))
    noise_x = jax.random.normal(prng_key,(128, 128))
    true_x = jax.random.normal(prng_key,(128, 128))

    denoise = transformer_ardm(true_x, noise_x, txt)
    assert denoise.shape == true_x.shape, f"Unexpected output shape from TransformerARDM: {denoise.shape}"

def test_save_load_consistency(dist_manager, prng_key, input_image):
    key = jax.random.split(prng_key, 2)
    
    txt = jnp.ones((128,))
    noise_x = jax.random.normal(prng_key,(128, 128))
    true_x = jax.random.normal(prng_key,(128, 128))
    
    transformer_ardm_initial = dad.TransformerARDM(dist_manager, key[0], res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)

    # Process input image
    denoise_initial = transformer_ardm_initial(true_x, noise_x, txt)

    # Save model using dist_manager
    sharding_initial = dist_manager.get_pytree_sharding(transformer_ardm_initial)
    dist_manager.save_pytree(transformer_ardm_initial, sharding_initial, "/test/transformer_ardm")

    # Reinstance model with new key
    transformer_ardm_reloaded = dad.TransformerARDM(dist_manager, key[1], res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)

    # Load model using dist_manager
    sharding_reloaded = dist_manager.get_pytree_sharding(transformer_ardm_reloaded)
    transformer_ardm_reloaded = dist_manager.load_pytree(sharding_reloaded, "/test/transformer_ardm")

    # Verify consistency after load
    denoise_reloaded = transformer_ardm_reloaded(true_x, noise_x, txt)

    # Check that the outputs after reload are consistent with the initial outputs
    assert jnp.allclose(denoise_initial, denoise_reloaded, atol=1e-5), "TransformerARDM outputs do not match after reload."

def test_transformer_ardm_causality(dist_manager, prng_key):
    transformer_ardm = dad.TransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=2, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
    
    key1, key2 = jax.random.split(prng_key)
    
    txt = jnp.ones((128,))
    noise_x = jax.random.normal(key1, (128, 128))
    true_x = jax.random.normal(key2, (128, 128))
    
    # Get the original output
    original_output = transformer_ardm((txt, true_x), noise_x, 0.0)
    
    # Modify the last element of true_x
    modified_true_x = true_x.at[-1].set(jnp.zeros_like(true_x[-1]))
    
    # Get the output with the modified input
    modified_output = transformer_ardm((txt, modified_true_x), noise_x, 0.0)
    
    # Check that only the last element of the output has changed
    assert jnp.allclose(original_output[:-1], modified_output[:-1]), "Non-causal behavior detected: earlier outputs changed"
    assert not jnp.allclose(original_output[-1], modified_output[-1]), "Last output should have changed"

def test_transformer_ardm_stability(dist_manager, prng_key):
    transformer_ardm = dad.TransformerARDM(dist_manager, prng_key, res_dim=128, 
            io_dim=128, vocab=128, n_layers=10, mlp_dim=256, qk_dim=128, v_dim=128, n_head=8)
    
    key1, key2 = jax.random.split(prng_key)
    
    txt = jnp.ones((128,))
    noise_x = jax.random.normal(key1, (128, 128))
    true_x = jax.random.normal(key2, (128, 128))
    
    # Run the model multiple times
    for _ in range(100):
        output = transformer_ardm((txt, true_x), noise_x, 0.0)
        
        # Check that the output is not NaN or inf
        assert not jnp.isnan(output).any(), "NaN values detected in the output"
        assert not jnp.isinf(output).any(), "Inf values detected in the output"
        
        # Check that the output magnitude is reasonable
        assert jnp.max(jnp.abs(output)) < 1e6, "Output magnitude is too large"
        
        # Use the output as the next input
        true_x = output
    
    # If we've made it this far without assertions, the test passes