import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import catfish.lvd.diffusion_core as dc

class NeuralModel(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear

    def __call__(self, x, z, neg_gamma):
        h = jnp.concatenate([z, neg_gamma[jnp.newaxis]])
        h = self.layer1(h)
        y = self.layer2(jax.nn.relu(h))
        return y 

@pytest.fixture
def setup_mixture_data():
    key = jax.random.PRNGKey(42)
    n_samples = 2048 
    
    mean1 = jnp.array([1, 1], dtype=jnp.float32)
    mean2 = jnp.array([-2, 1], dtype=jnp.float32)
    std1 = jnp.array([0.5, 1.5], dtype=jnp.float32)
    std2 = jnp.array([1, 1.5], dtype=jnp.float32)
    # Mixing ratio
    mix_ratio = 0.5
    s1 = int(n_samples * mix_ratio)
    s2 = n_samples - int(n_samples * mix_ratio)

    # Generate mixed Gaussian data
    keys = jax.random.split(key, 3)
    samples1 = std1 * jax.random.normal(keys[0], (s1, 2)) + mean1
    samples2 = std2 * jax.random.normal(keys[1], (s2, 2)) + mean2
    y_data = jnp.concatenate([samples1, samples2], axis=0)
    x_data = y_data*0
    return x_data, y_data, keys[2]

@pytest.fixture
def setup_model():
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 2)
    model = NeuralModel(
        layer1=eqx.nn.Linear(3, 2048, key=keys[0]),  # input dimension is 10 because x and z are concatenated
        layer2=eqx.nn.Linear(2048, 2, key=keys[1])    # output the same dimension as input
    )
    return model

def test_mixture_model(setup_mixture_data, setup_model):
    x_data, y_data, key = setup_mixture_data
    model = setup_model
    data = (x_data, y_data)
    n_steps = 100
    shape = (2,)

    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(model)
    
    state = (model, opt_state, key)

    def loss_fn(model, data, key):
        loss = dc.diffusion_loss(model , data, dc.f_neg_gamma, key)
        return loss

    # Training loop
    for _ in range(250000):  # Enough iterations to ensure convergence
        loss, state = dc.update_state(state, data, optimizer, loss_fn)

    # Sampling
    sampled_output = dc.sample_diffusion(x_data, state[0], dc.f_neg_gamma, key, n_steps, shape) 

    # Verification: Check moments along random directions
    directions = jax.random.normal(key, (3, 2))  # Three random projections
    projected_original = jnp.dot(y_data, directions.T)
    projected_sampled = jnp.dot(sampled_output, directions.T)


    for i in range(3):
        print(jnp.mean(projected_original[:, i]), jnp.mean(projected_sampled[:, i]))
        assert jnp.allclose(jnp.mean(projected_original[:, i]), jnp.mean(projected_sampled[:, i]), atol=0.15), "Means do not match"
        print(jnp.var(projected_original[:, i]), jnp.var(projected_sampled[:, i]))
        assert jnp.allclose(jnp.var(projected_original[:, i]), jnp.var(projected_sampled[:, i]), atol=0.15), "Variances do not match"
