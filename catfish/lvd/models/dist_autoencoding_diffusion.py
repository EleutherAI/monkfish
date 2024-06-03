import os

import jax
import jax.numpy as jnp
import equinox as eqx

import catfish.lvd.models.dist_layers as dl

def reshape_to_patches(data, patch_width=16, patch_height=8):
    num_channels, width, height = data.shape
    
    assert height % patch_height == 0
    assert width % patch_width == 0

    height_patches = height // patch_height
    width_patches = width // patch_width
    patch_channels = num_channels * patch_width * patch_height
    
    reshaped_data = data.reshape(num_channels, width_patches, patch_width, height_patches, patch_height)
    transposed_data = reshaped_data.transpose(0, 4, 2, 3, 1)
    patches = transposed_data.reshape(patch_channels, height_patches, width_patches)
    
    return patches

def reconstruct_from_patches(patches, patch_width=16, patch_height=8):

    patch_channels, height_patches, width_patches = patches.shape
   
    assert patch_channels % (patch_height * patch_width) == 0

    num_channels = patch_channels // (patch_height * patch_width)
    height = height_patches*patch_height
    width = width_patches*patch_width
    
    transposed_data = patches.reshape(num_channels, patch_height, patch_width, height_patches, width_patches)
    reshaped_data = transposed_data.transpose(0, 4, 2, 3, 1)
    data = reshaped_data.reshape(num_channels, width, height)
    
    return data

class Encoder(eqx.Module):
    layers: list
    
    def __init__(self, dist_manager, key,  k, n_layers):
        keys = jax.random.split(key,n_layers + 2)
        
        layers = []
        
        embedding = dl.ShrdConv(dist_manager, keys[0], 1,1, 384,128*k)
        layers.append(embedding)
        for i in range(n_layers):
            res_block = dl.ConvResBlock(dist_manager, keys[0], 128*k, 4*128*k)
            layers.append(res_block)
        unembedding = dl.ShrdConv(dist_manager, keys[-1], 1,1, 128*k, 32)
        layers.append(unembedding)
        
        self.layers = layers

    
    #[3 x 512 x 256] -> [32 x 32 x 32]
    def __call__(self, x):
        h = reshape_to_patches(x)
        for i in range(0,len(self.layers)):
            print(self.layers[i])
            h = self.layers[i](h)
        
        y = h
        return y

    def save(self, path_prefix):
        for i,layer in enumerate(self.layers):
            path = os.path.join(path_prefix,f"encoder_layer_{i}")
            layer.save(path)
    
    def load(self, path_prefix):
        new_self = self
        
        for i in range(len(self.layers)):
            path = os.path.join(path_prefix, f"encoder_layer_{i}")
            layer = self.layers[i].load(path)
            where = lambda x: x.layers[i]
            new_self = eqx.tree_at(where, new_self, layer)

        return new_self


class Decoder(eqx.Module):
    #[3 x 512 x 256] x [32 x 32 x 32] -> [3 x 512 x 256]
    layers: list
    decode_embed:  dl.ShrdConv
    
    def __init__(self, dist_manager, key, k, n_layers):
        keys = jax.random.split(key,n_layers + 3)
        
        layers = []
        
        embedding = dl.ShrdConv(dist_manager, keys[0], 1, 1, 384, 128*k)
        layers.append(embedding)
        for i in range(n_layers):
            res_block = dl.ConvResBlock(dist_manager, keys[0], 128*k, 4*128*k)
            layers.append(res_block)
        unembedding = dl.ShrdConv(dist_manager, keys[-2], 1, 1, 128*k, 384)
        layers.append(unembedding)
        
        self.layers = layers
        self.decode_embed = dl.ShrdConv(dist_manager, keys[-1], 1, 1, 32,128*k)

    def __call__(self, x, embed):
        x_patches = reshape_to_patches(x)
        h = self.layers[0](x_patches) + self.decode_embed(embed)
        for i in range(1,len(self.layers)):
            h = self.layers[i](h)
        
        y_patches = h
        y = reconstruct_from_patches(y_patches)
        return y

    def save(self, path_prefix):
        for i,layer in enumerate(self.layers):
            path = os.path.join(path_prefix,f"decoder_layer_{i}")
            layer.save(path)
        
        path = os.path.join(path_prefix,f"decoder_unembed")
        self.decode_embed.save(path)
    
    def load(self, path_prefix):
        new_self = self
        
        for i in range(len(self.layers)):
            path = os.path.join(path_prefix, f"decoder_layer_{i}")
            layer = self.layers[i].load(path)
            where = lambda x: x.layers[i]
            new_self = eqx.tree_at(where, new_self, layer)
        
        path = os.path.join(path_prefix,f"decoder_unembed")
        decode_embed = self.decode_embed.load(path)
        where = lambda x: x.decode_embed
        new_self = eqx.tree_at(where, new_self, decode_embed)

        return new_self
    

