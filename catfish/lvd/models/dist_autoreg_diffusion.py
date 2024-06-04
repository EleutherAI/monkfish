import os

import jax
import jax.numpy as jnp
import equinox as eqx

import catfish.lvd.models.dist_layers as dl

class TransformerARDM(eqx.Module):
    
    layers: list
    true_x_enc:  dl.ShrdLinear
    noise_x_enc:  dl.ShrdLinear
    txt_enc:  dl.ShrdLinear
    x_decode:  dl.ShrdLinear
    
    def __init__(self, dist_manager, key, k, n_layers):
        keys = jax.random.split(key,n_layers + 3)
        
        layers = []
        
        embedding = dl.ShrdConv(dist_manager, keys[0], 1, 1, 384, 128*k)
        layers.append(embedding)
        for i in range(n_layers):
            res_block = dl.ShrdMHAttention(dist_manager, keys[i], 128*k, 4*128*k)
            layers.append(res_block)
        unembedding = dl.ShrdConv(dist_manager, keys[-2], 1, 1, 128*k, 384)
        layers.append(unembedding)
        
        self.layers = layers
        self.decode_embed = dl.ShrdConv(dist_manager, keys[-1], 1, 1, 32,128*k)

    #[txt_pos] x [x_pos x d_io] x [x_pos x d_io] -> [x_pos x d_io]
    def __call__(self, true_x, noise_x, txt):
        h_suffix = (jax.vmap(self.true_x_enc)(true_x) +
            jax.vmap(self.noise_x_enc)(noise_x))
        h_prefix = jax.vmap(self.txt_enc)(txt)
        h = jnp.concatenate([h_prefix, h_suffix], axis=0)

        for i in range(1,len(self.layers)):
            h = self.layers[i](h)
        
        y = jax.vmap(self.x_decode)(h[len(h_suffix):])
        return y

    def save(self, path_prefix):
        for i,layer in enumerate(self.layers):
            path = os.path.join(path_prefix,f"layer_{i}")
            layer.save(path)
        
        path = os.path.join(path_prefix,f"true_x_enc")
        self.true_x_enc.save(path)
        
        path = os.path.join(path_prefix,f"noise_x_enc")
        self.noise_x_enc.save(path)
        
        path = os.path.join(path_prefix,f"txt_enc")
        self.txt_enc.save(path)
        
        path = os.path.join(path_prefix,f"x_decode")
        self.x_decode.save(path)
    
    def load(self, path_prefix):
        new_self = self
        
        for i in range(len(self.layers)):
            path = os.path.join(path_prefix, f"decoder_layer_{i}")
            layer = self.layers[i].load(path)
            where = lambda x: x.layers[i]
            new_self = eqx.tree_at(where, new_self, layer)
    
        #TODO: rewrite
        path = os.path.join(path_prefix,f"true_x_enc")
        decode_embed = self.true_x_enc.load(path)
        where = lambda x: x.true_x_enc
        new_self = eqx.tree_at(where, new_self, decode_embed)
        
        path = os.path.join(path_prefix,f"noise_x_enc")
        decode_embed = self.noise_x_enc.load(path)
        where = lambda x: x.noise_x_enc
        new_self = eqx.tree_at(where, new_self, decode_embed)
        
        path = os.path.join(path_prefix,f"txt_enc")
        decode_embed = self.txt_enc.load(path)
        where = lambda x: x.txt_enc
        new_self = eqx.tree_at(where, new_self, decode_embed)
        
        path = os.path.join(path_prefix,f"x_decode")
        decode_embed = self.x_decode.load(path)
        where = lambda x: x.x_decode
        new_self = eqx.tree_at(where, new_self, decode_embed)

        return new_self