
class TransformerARDM(eqx.Module):
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