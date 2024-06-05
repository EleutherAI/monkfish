import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import catfish.lvd.diffusion_core as dc

class NeuralModel(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    def __init__:

    def __call__(self, x, z, neg_gamma):
        h = self.layer1(jnp.concatenate([z, neg_gamma]))
        y = self.layer2(jax.nn.relu(h))
        return y 

@pytest.fixture
def setup_mixture_data():
    key = jax.random.PRNGKey(42)
    n_samples = 256 
    
    mean1 = [1, 1]
    mean2 = [-2, 1]
    std1 = [0.5, 1.5]
    std2 = [1, 1.5]
    # Mixing ratio
    mix_ratio = 0.5
    s1 = int(n_samples * mix_ratio)
    s2 = n_samples - int(n_samples * mix_ratio)

    # Generate mixed Gaussian data
    keys = jax.random.split(key, 3)
    samples1 = std1 * jax.random.normal(keys[0], (s1, 2)) + mean1
    samples2 = std2 * jax.random.normal(keys[1], (s2, 2)) + mean2
    x_data = jnp.concatenate([samples1, samples2], axis=0)
    y_data = x_data*0
    return x_data, y_data, keys[2]

@pytest.fixture
def setup_model():
    model = NeuralModel(
        layer1=eqx.nn.Linear(3, 256),  # input dimension is 10 because x and z are concatenated
        layer2=eqx.nn.Linear(256, 5)    # output the same dimension as input
    )
    return model

def test_mixture_model(setup_mixture_data, setup_model):
    x_data, y_data, key = setup_mixture_data
    model = setup_model
    data = (x_data, y_data)
    n_steps = 100
    shape = (2,)

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(model)
    
    state = (model, opt_state, key)

    # Training loop
    for _ in range(1000):  # Enough iterations to ensure convergence
        loss, state = dc.update_state(state, data, optimizer, dc.diffusion_loss)
        print(loss)

    # Sampling
    sampled_output = dc.sample_diffusion(x_data, state[0], dc.f_neg_gamma, key, n_steps, shape)

    # Verification: Check moments along random directions
    directions = jax.random.normal(key, (3, 2))  # Three random projections
    projected_original = jnp.dot(x_data, directions.T)
    projected_sampled = jnp.dot(sampled_output, directions.T)

    for i in range(3):
        assert jnp.allclose(jnp.mean(projected_original[:, i]), jnp.mean(projected_sampled[:, i]), atol=0.5), "Means do not match"
        assert jnp.allclose(jnp.var(projected_original[:, i]), jnp.var(projected_sampled[:, i]), atol=0.5), "Variances do not match"

if __name__ == "__main__":
    pytest.main([__file__])
