import optax

import catfish.lvd.models.dist_autoencoding_diffusion as daed
import catfish.lvd.models.dist_utils as du

class DiffAEHarness:
    """Sharded Diffusion autoencoder harness"""

    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.model = None
        self.opt_state = None
        self.optimizer = None
        
        self.dist_manager = du.DistManager(conf)

        self.init_dist_manager()
        self.make_model()
    
    def init_dist_manager():
    
    def make_model(self):
        args = self.args

        model_conf = self.conf["diffusion_auto_encoder"]["model"]
        enc_conf = model_conf["encoder"]
        dec_conf = model_conf["decoder"]
        
        self.model = {
            "encoder": daed.Encoder(
                dist_manager, 
                key=, 
                k =enc_conf["k"],
                n_layers=enc_conf["n_layers"],
            ),
            "decoder": daed.Decoder(

            )
        }

    
    def make_optimizer(self, model):
        args = self.args
        cfg = self.cfg
        
        optimizer = optax.adam()
    
    def train(self):
        args = self.args
        cfg = self.cfg

    def autoencode(self):
        args = self.args
        cfg = self.cfg
    



