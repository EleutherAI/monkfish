import os

import jax
import jax.numpy as jnp
import equinox as eqx

import monkfish.lvd.models.dist_layers as dl
import monkfish.lvd.models.dist_attn_layers as dal

class SplitTransformerARDM(eqx.Module):
    #TransformerARDM with split conditioning of io streams
    #Enabling separate noise levels for each reconstruction target 
    
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
            res_block = dal.SplitTransformerBlock(
                dist_manager, keys[i], res_dim,
                mlp_dim, qk_dim, v_dim, n_head)
            layers.append(res_block)
        
        self.layers = layers
        
        self.true_x_enc = dl.ShrdLinear(dist_manager, keys[-4], io_dim, res_dim)
        self.noise_x_enc = dl.ShrdLinear(dist_manager, keys[-3], io_dim, res_dim)
        self.txt_enc = dl.ShrdLinear(dist_manager, keys[-2], vocab, res_dim)
        self.x_decode = dl.ShrdLinear(dist_manager, keys[-1], res_dim, io_dim)

    #Mode == "train"
    #([txt_pos] x [x_pos x d_io]) x [x_pos x d_io] x [] -> [x_pos x d_io]
    #Mode == "generate"
    #([txt_pos] x [x_pos x d_io]) x [(x_pos+1) x d_io] x [] -> [x_pos+1 x d_io]
    def __call__(self, data, noise_x, neg_gamma, mode="train"):
        assert mode in ["train","generate"]
        txt, true_x = data
        
        assert txt.shape[0] >= 1
        if mode == "train":
            assert noise_x.shape[0] == true_x.shape[0]
        elif mode == "generate":
            assert noise_x.shape[0] == true_x.shape[0]+1
        
        vocab = self.txt_enc.weight.shape[0]

        #Scale text input by 1/sqrt(vocab) to normalize input magnitude
        h_prefix = jax.vmap(self.txt_enc)(jax.nn.one_hot(txt, vocab)*jnp.sqrt(vocab))
        h_suffix = jax.vmap(self.true_x_enc)(true_x)
        h_inp = jnp.concatenate([h_prefix, h_suffix], axis=0)

        h_noise = jnp.zeros((txt.shape[0] + true_x.shape[0], h_inp.shape[1]))

        if mode == "train":
            h_noise_slice = h_noise.at[txt.shape[0]-1:-1]
        elif mode == "generate":
            h_noise_slice = h_noise.at[txt.shape[0]-1:]

        h_noise = h_noise_slice.set(
            jax.vmap(self.noise_x_enc)(noise_x))

        h_noise.at[:,0].set(h_noise[:,0] + neg_gamma/10)
        
        h = (h_inp,h_noise)
        #print("A",h[0][-3:,5],h[1][-3:,5])
        

        layer_scale = 1/len(self.layers)
        for i in range(len(self.layers)):
            h_diff = self.layers[i](h)
            #print("B",h_diff[0][-3:,5],h_diff[1][-3:,5])
            h = (
                (h[0] + h_diff[0]*layer_scale),
                (h[1] + h_diff[1]*layer_scale)
            )
            #print("C",h[0][-3:,5],h[1][-3:,5])
        
        if mode == "train":
            y = jax.vmap(self.x_decode)(h[1][txt.shape[0]-1:-1])
            return y
        elif mode == "generate":
            y = jax.vmap(self.x_decode)(h[1][txt.shape[0]-1:])
            return y

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
            res_block = dal.TransformerBlock(
                dist_manager, keys[i], res_dim,
                mlp_dim, qk_dim, v_dim, n_head)
            layers.append(res_block)
        
        self.layers = layers
        
        self.true_x_enc = dl.ShrdLinear(dist_manager, keys[-4], io_dim, res_dim)
        self.noise_x_enc = dl.ShrdLinear(dist_manager, keys[-3], io_dim, res_dim)
        self.txt_enc = dl.ShrdLinear(dist_manager, keys[-2], vocab, res_dim)
        self.x_decode = dl.ShrdLinear(dist_manager, keys[-1], res_dim, io_dim)

    #Mode == "train"
    #([txt_pos] x [x_pos x d_io]) x [x_pos x d_io] x [] -> [x_pos x d_io]
    #Mode == "generate"
    #([txt_pos] x [x_pos x d_io]) x [(x_pos+1) x d_io] x [] -> [x_pos+1 x d_io]
    def __call__(self, data, noise_x, neg_gamma, mode="train"):
        assert mode in ["train","generate"]
        txt, true_x = data
        
        assert txt.shape[0] >= 1
        if mode == "train":
            assert noise_x.shape[0] == true_x.shape[0]
        elif mode == "generate":
            assert noise_x.shape[0] == true_x.shape[0]+1
        
        vocab = self.txt_enc.weight.shape[0]

        #Scale text input by 1/sqrt(vocab) to normalize input magnitude
        h_prefix = jax.vmap(self.txt_enc)(jax.nn.one_hot(txt, vocab)*jnp.sqrt(vocab))
        h_suffix = jax.vmap(self.true_x_enc)(true_x)
        h_inp = jnp.concatenate([h_prefix, h_suffix], axis=0)

        h_noise = jnp.zeros((txt.shape[0] + true_x.shape[0], h_inp.shape[1]))

        if mode == "train":
            h_noise_slice = h_noise.at[txt.shape[0]-1:-1]
        elif mode == "generate":
            h_noise_slice = h_noise.at[txt.shape[0]-1:]

        h_noise = h_noise_slice.set(
            jax.vmap(self.noise_x_enc)(noise_x))

        h = h_noise + h_inp
        
        #TODO: make neg_gamma_conditionining less hacky
        h.at[:,0].set(h[:,0] + neg_gamma/10)

        layer_scale = 1/len(self.layers)
        for i in range(len(self.layers)):
            h = h + self.layers[i](h)*layer_scale
        
        if mode == "train":
            y = jax.vmap(self.x_decode)(h[txt.shape[0]-1:-1])
            return y
        elif mode == "generate":
            y = jax.vmap(self.x_decode)(h[txt.shape[0]-1:])
            return y