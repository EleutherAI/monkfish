import os

import jax
import jax.numpy as jnp
import equinox as eqx

import monkfish.lvd.models.dist_layers as dl

class TransformerARDM(eqx.Module):
    
    layers: list
    true_x_enc:  dl.ShrdLinear
    noise_x_enc:  dl.ShrdLinear
    txt_enc:  dl.ShrdLinear
    x_decode:  dl.ShrdLinear
    
    def __init__(self, dist_manager, key, res_dim, 
                 io_dim, vocab, n_layers,
                 mlp_dim, qk_dim, v_dim, n_head):
        keys = jax.random.split(key,n_layers + 4)
        
        layers = []
        
        for i in range(n_layers):
            res_block = dl.TransformerBlock(
                dist_manager, keys[i], res_dim,
                mlp_dim, qk_dim, v_dim, n_head)
            layers.append(res_block)
        
        self.layers = layers
        
        self.true_x_enc = dl.ShrdLinear(dist_manager, keys[-4], io_dim, res_dim)
        self.noise_x_enc = dl.ShrdLinear(dist_manager, keys[-3], io_dim, res_dim)
        self.txt_enc = dl.ShrdLinear(dist_manager, keys[-2], vocab, res_dim)
        self.x_decode = dl.ShrdLinear(dist_manager, keys[-1], res_dim, io_dim)

    #([txt_pos] x [x_pos x d_io]) x [x_pos x d_io] x [] -> [x_pos x d_io]
    def __call__(self, data, noise_x, neg_gamma):
        txt, true_x = data
        vocab = self.txt_enc.weight.shape[0]

        h_suffix = (jax.vmap(self.true_x_enc)(true_x) +
            jax.vmap(self.noise_x_enc)(noise_x))
        h_prefix = jax.vmap(self.txt_enc)(jax.nn.one_hot(txt, vocab))
        h = jnp.concatenate([h_prefix, h_suffix], axis=0)
        
        #TODO: make neg_gamma_conditionining less hacky
        h.at[:,0].set(h[:,0] + neg_gamma)

        for i in range(1,len(self.layers)):
            h = self.layers[i](h)
        
        y = jax.vmap(self.x_decode)(h[len(h_suffix):])
        return y