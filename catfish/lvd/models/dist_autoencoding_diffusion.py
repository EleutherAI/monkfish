import os

import jax
import equinox as eqx

import catfish.lvd.models.dist_layers as dl

def reshape_to_patches(data, patch_height=8, patch_width=16):
    num_channels, height, width = data.shape
    # Ensure the data dimensions are divisible by patch dimensions
    assert height % patch_height == 0
    assert width % patch_width == 0
    
    reshaped_data = data.reshape(num_channels, height // patch_height, patch_height, width // patch_width, patch_width)
    transposed_data = reshaped_data.transpose(0, 2, 4, 1, 3)
    patches = transposed_data.reshape(-1, height // patch_height, width // patch_width)
    
    return patches

def reconstruct_from_patches(patches, original_shape, patch_height=8, patch_width=16):
    num_channels, height, width = original_shape
    
    reshaped_patches = patches.reshape(num_channels, patch_height, patch_width, height // patch_height, width // patch_width)
    transposed_patches = reshaped_patches.transpose(0, 3, 1, 4, 2)
    data = transposed_patches.reshape(original_shape)
    
    return data

class Encoder(eqx.Module):
    layers: list
    
    def __init__(self, dist_manager, key,  k, n_layers):
        keys = jax.random.split(key,n_layers + 2)
        
        layers = []
        
        embedding = dl.ShrdConv(dist_manager, keys[0], 1,1,384,128*k)
        layers.append(embedding)
        for i in range(n_layers):
            res_block = dl.ConvResBlock(dist_manager, keys[0],128*k)
            layers.append(res_block)
        unembedding = dl.ShrdConv(dist_manager, keys[-1], 1,1,128*k,32)
        layers.append(unembedding)
        
        self.layers = layers

    
    #[3 x 512 x 256] -> [32 x 32 x 32]
    def __call__(self, x):
        h = reshape_to_patches(x)
        for i in range(0,len(self.layers)):
            h = self.layers[i](h)
        
        y_patches = h
        y = reconstruct_from_patches(y_patches)
        return y

    def save(self, path_prefix):
        for i,layer in enumerate(self.layers):
            path = os.path.join(path_prefix,f"encoder_layer_{i}")
            layer.save(path)
    
    def load(self, path_prefix):
        for i,layer in enumerate(self.layers):
            path = os.path.join(path_prefix, f"encoder_layer_{i}")
            layer.load(path)


class Decoder(eqx.Module):
    #[3 x 512 x 256] x [32 x 32 x 32] -> [512 x 256 x 3]
    layers: list
    decode_embed:  dl.ShrdConv
    
    def __init__(self, dist_manager, key, k, n_layers):
        keys = jax.random.split(key,n_layers + 3)
        
        layers = []
        
        embedding = dl.ShrdConv(dist_manager, keys[0], 1, 1, 384, 128*k)
        layers.append(embedding)
        for i in range(n_layers):
            res_block = dl.ConvResBlock(dist_manager, keys[0], 128*k)
            layers.append(res_block)
        unembedding = dl.ShrdConv(dist_manager, keys[-2], 1, 1, 128*k, 384)
        layers.append(unembedding)
        
        self.layers = layers
        self.decode_embed = dl.ShrdConv(dist_manager, keys[-1], 1, 1, 32,128*k)

    def __call__(self, x, embed):
        x_patches = reshape_to_patches(x)
        h = self.layers[0](x_patches) + self.decode_embed(embed)
        for i in range(1,len(layers)):
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
        for i,layer in enumerate(self.layers):
            path = os.path.join(path_prefix, f"decoder_layer_{i}")
            layer.load(path)
        
        path = os.path.join(path_prefix,f"decoder_unembed")
        self.decode_embed.load(path)


