import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

import catfish.lvd.diffusion_core as dc

# A simple model for testing purposes
class SimpleModel(eqx.Module):
    weights: jnp.ndarray

    def __call__(self, x, z, neg_gamma):
        # Simplified model just doing linear transformation for the sake of testing
        return jnp.dot(z, self.weights) + x

@pytest.fixture
def setup_data():
    key = jax.random.PRNGKey(0)
    x_data = jax.random.normal(key, (10, 5))
    y_data = jax.random.normal(key, (10, 5))
    return x_data, y_data, key

@pytest.fixture
def setup_model():
    weights = jax.random.normal(jax.random.PRNGKey(1), (5,))
    model = SimpleModel(weights=weights)
    return model

def test_diffusion_loss(setup_data, setup_model):
    x_data, y_data, key = setup_data
    model = setup_model
    loss = dc.diffusion_loss(model, (x_data, y_data), dc.f_neg_gamma, key)
    assert loss.shape == ()

def test_sample_diffusion(setup_model, setup_data):
    model = setup_model
    x_data, _, key = setup_data
    sampled_output = dc.sample_diffusion(x_data, model, dc.f_neg_gamma, key, 10, (5,))
    assert sampled_output.shape == (10, 5)

def test_end_to_end(setup_data, setup_model):
    x_data, y_data, key = setup_data
    model = setup_model
    data = (x_data, y_data)
    n_steps = 10
    shape = (5,)

    # Simulate a training step
    optimizer = eqx.optim.Adam(0.01)
    opt_state = optimizer.init(model)
    state = (model, opt_state, key)
    loss, new_state = dc.update_state(state, data, optimizer, dc.diffusion_loss)

    # Test if training reduces loss (very basic check)
    loss_after, _ = update_state(new_state, data, optimizer, dc.diffusion_loss)
    assert loss_after < loss

    # Test sampling
    sampled_output = sample_diffusion(x_data, new_state[0], dc.f_neg_gamma, key, n_steps, shape)
    assert sampled_output.shape == (10, 5)

