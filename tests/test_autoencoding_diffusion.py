import pytest
import jax
import jax.numpy as jnp

import catfish.lvd.models.dist_layers as dl
import catfish.lvd.models.dist_utils as du
import catfish.lvd.models.dist_autoencoding_diffusion as dad

def test_encoder_decoder_initialization(dist_manager, prng_key):
    key = jax.random.split(prng_key, 2)
    encoder = dad.Encoder(dist_manager, key[0], k=2, n_layers=2)
    decoder = dad.Decoder(dist_manager, key[1], k=2, n_layers=2)
    assert isinstance(encoder, dad.Encoder), "Encoder initialization failed"
    assert isinstance(decoder, dad.Decoder), "Decoder initialization failed"

def test_encoder_decoder_forward_pass(dist_manager, prng_key, input_image):
    key = jax.random.split(prng_key, 2)
    encoder = dad.Encoder(dist_manager, key[0], k=2, n_layers=2)
    decoder = dad.Decoder(dist_manager, key[1], k=2, n_layers=2)

    encoded = encoder(input_image)
    # Create a dummy neg_gamma value
    neg_gamma = jnp.array(0.5)
    decoded = decoder(encoded, input_image, neg_gamma)
    assert encoded.shape == (32, 32, 32), f"Unexpected output shape from encoder: {encoded.shape}"
    assert decoded.shape == input_image.shape, f"Output shape mismatch in decoder, expected {input_image.shape}, got {decoded.shape}"

def test_save_load_consistency(dist_manager, prng_key, input_image):
    key = jax.random.split(prng_key, 2)
    key_initial = jax.random.split(key[0], 3)
    encoder_initial = dad.Encoder(dist_manager, key_initial[0], k=2, n_layers=2)
    decoder_initial = dad.Decoder(dist_manager, key_initial[1], k=2, n_layers=2)

    # Process input image
    encoded_initial = encoder_initial(input_image)
    # Create a dummy neg_gamma value
    neg_gamma = jnp.array(0.5)
    decoded_initial = decoder_initial(encoded_initial, input_image, neg_gamma)

    # Save models using dist_manager
    encoder_sharding = dist_manager.get_pytree_sharding(encoder_initial)
    decoder_sharding = dist_manager.get_pytree_sharding(decoder_initial)
    dist_manager.save_pytree(encoder_initial, encoder_sharding, "/test/encoder")
    dist_manager.save_pytree(decoder_initial, decoder_sharding, "/test/decoder")

    # Reinstance models with new keys
    key_reloaded = jax.random.split(key[1], 3)
    encoder_reloaded = dad.Encoder(dist_manager, key_reloaded[0], k=2, n_layers=2)
    decoder_reloaded = dad.Decoder(dist_manager, key_reloaded[1], k=2, n_layers=2)

    # Load models using dist_manager
    encoder_new_sharding = dist_manager.get_pytree_sharding(encoder_reloaded)
    decoder_new_sharding = dist_manager.get_pytree_sharding(decoder_reloaded)
    encoder_reloaded = dist_manager.load_pytree(encoder_new_sharding, "/test/encoder")
    decoder_reloaded = dist_manager.load_pytree(decoder_new_sharding, "/test/decoder")


def test_patch_reconstruction_integrity(input_image):
    patches = dad.reshape_to_patches(input_image)
    reconstructed = dad.reconstruct_from_patches(patches)
    assert jnp.allclose(input_image, reconstructed), "Patch reconstruction failed to retain data integrity."

def test_patch_contiguity(input_image):
    transform_image = input_image.at[:,:16,:8].set(1)
    patches = dad.reshape_to_patches(transform_image, patch_width=16, patch_height=8)
    assert jnp.allclose(jnp.mean(patches[:,0,0]),1)
