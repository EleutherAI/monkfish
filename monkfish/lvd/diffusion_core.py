import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

import functools

#Linear SNR Schedule
def f_neg_gamma(t, min_snr= -10, max_snr = 10):
    #equivalent to log SNR
    return max_snr - t*(max_snr - min_snr)

def sigma_squared(neg_gamma):
    return jax.nn.sigmoid(-neg_gamma)

def alpha_squared(neg_gamma):
    return jax.nn.sigmoid(neg_gamma)

def diffusion_loss(model, data, f_neg_gamma,  key):
    #As defined in https://arxiv.org/abs/2107.00630 eq. #17 
    x_data, y_data = data

    batch_size = y_data.shape[0]
    # Check if x_data is a pytree or an array
    if isinstance(x_data, jnp.ndarray):
        # If it's an array or a simple container, check its first dimension
        assert x_data.shape[0] == batch_size, "x_data first dimension doesn't match y_data"
    else:
        # If it's a pytree, check the first dimension of each leaf
        def check_shape(leaf):
            assert leaf.shape[0] == batch_size, f"Leaf shape {leaf.shape} doesn't match y_data shape {y_data.shape}"
            return leaf
        
        jtu.tree_map(check_shape, x_data)
    
    batch_size = y_data.shape[0]
    
    keys = jax.random.split(key, batch_size)

    def _diffusion_loss(model, f_neg_gamma, x_data, y_data, key):

        t_key, noise_key = jax.random.split(key,2)
        
        t = jax.random.uniform(t_key)
        
        neg_gamma, neg_gamma_prime = jax.value_and_grad(f_neg_gamma)(t)

        alpha, sigma = jnp.sqrt(alpha_squared(neg_gamma)), jnp.sqrt(sigma_squared(neg_gamma))

        epsilon = jax.random.normal(noise_key, shape = y_data.shape)

        z = y_data*alpha + sigma*epsilon

        epsilon_hat = model(x_data, z, neg_gamma)

        loss = -1/2*neg_gamma_prime*(epsilon_hat-epsilon)**2

        return jnp.sum(loss)

    losses = jax.vmap(lambda x, y, z: _diffusion_loss(model, f_neg_gamma, x, y, z))(x_data, y_data, keys)
    mean_loss = jnp.sum(losses)/y_data.size

    return mean_loss

def sample_diffusion(inputs, model, f_neg_gamma, key, n_steps, shape):
    #Following https://arxiv.org/abs/2202.00512 eq. #8
    time_steps = jnp.linspace(0, 1, num=n_steps+1)

    n_samples = inputs.shape[0]

    z = jax.random.normal(key, (n_samples,) + shape)
    for i in range(n_steps):
        # t_s < t_t
        t_s, t_t = time_steps[n_steps-i-1], time_steps[n_steps-i]

        neg_gamma_s, neg_gamma_t = f_neg_gamma(t_s), f_neg_gamma(t_t)
        
        alpha_s = jnp.sqrt(alpha_squared(neg_gamma_s))
        alpha_t, sigma_t = jnp.sqrt(alpha_squared(neg_gamma_t)), jnp.sqrt(sigma_squared(neg_gamma_t))

        epsilon_hat = jax.vmap(lambda x, y: model(x, y, neg_gamma_t))(inputs, z)

        k = jnp.exp((neg_gamma_t-neg_gamma_s)/2)
        z = (alpha_s/alpha_t)*(z + sigma_t*epsilon_hat*(k-1))

    outputs = z

    return outputs

@functools.partial(jax.jit, static_argnums=(2, 3))
def update_state(state, data, optimizer, loss_fn):
    model, opt_state, key = state
    new_key, subkey = jax.random.split(key)
    
    loss, grads = jax.value_and_grad(loss_fn)(model, data, subkey)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    new_state = new_model, new_opt_state, new_key
    
    return loss,new_state

@functools.partial(jax.jit, static_argnums=(2, 3))
def update_state_dict(state_dict, data, optimizer, loss_fn):
    state = (
        state_dict["model"],
        state_dict["opt_state"],
        state_dict["prng_key"]
    )
    loss, new_state = update_state(state, data, optimizer, loss_fn)
    new_state_dict = {
        "model": new_state[0],
        "opt_state": new_state[1],
        "prng_key": new_state[2]
    }
    return loss, new_state_dict
