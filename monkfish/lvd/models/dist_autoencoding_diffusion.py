import os

import jax
import jax.numpy as jnp
import equinox as eqx

import monkfish.lvd.models.dist_layers as dl

def reshape_to_patches(data, patch_width=16, patch_height=8):
    "take a large image and convert chunks of it into channels"
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
    """take an image converted into channels and convert it back
    into a base image"""

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
            res_block = dl.ConvResBlock(dist_manager, keys[i], 128*k, 4*128*k)
            layers.append(res_block)
        unembedding = dl.ShrdConv(dist_manager, keys[-1], 1,1, 128*k, 32)
        layers.append(unembedding)
        
        self.layers = layers

    
    #[3 x 512 x 256] -> [32 x 32 x 32]
    def __call__(self, x):
        h = reshape_to_patches(x)
        for i in range(0,len(self.layers)):
            h = self.layers[i](h)
        
        y = h
        return y

class Decoder(eqx.Module):
    #[32 x 32 x 32] x [3 x 512 x 256] x [] -> [3 x 512 x 256]
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

    def __call__(self, embed, x, neg_gamma):
        x_patches = reshape_to_patches(x)
        h = self.layers[0](x_patches) + self.decode_embed(embed)
        #TODO: make neg_gamma_conditionining less hacky
        h.at[0].set(h[0] + neg_gamma)
        for i in range(1,len(self.layers)):
            h = self.layers[i](h)
        
        y_patches = h
        y = reconstruct_from_patches(y_patches)
        return y