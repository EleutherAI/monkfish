import os
import re
import collections

import fs
import jax
import optax

import catfish.lvd.models.dist_autoencoding_diffusion as daed
import catfish.lvd.models.dist_utils as du
import catfish.lvd.shrd_data_loader as sdl
import catfish.lvd.diffusion_core as dc


DAEModel = collections.namedtuple(["encoder", "decoder"])

class DiffAEHarness:
    """Sharded Diffusion autoencoder harness"""

    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.state = {}
        self.optimizer = None
        self.prng_key = None
        self.dist_manager = None
        self.data_loader = None
        self.credentials_path = None
        self.data_fs = None
        self.ckpt_fs = None

        self.parse_args()
        self.init_fs()
        self.init_dist_manager()
        self.init_data_loader()
        self.make_model()

    def init_fs(self):
        gcp_conf = self.cfg["gcp"]
        gcp_credentials_path =  gcp_conf["gcp_credentials_path"]
        gcp_bucket_name =  gcp_conf["gcp_bucket_name"]

        #Initialize data loader file system
        dl_conf = self.cfg["diffusion_auto_encoder"]["data_loader"]
        dl_fs_type = dl_conf["fs_type"]
        dl_root_directory = dl_conf["data_root_directory"]

        if dl_fs_type == "local":
            self.data_fs = sdl.os_filesystem(dl_root_directory)
        elif dl_fs_type == "gcp":
            self.data_fs = sdl.gcp_filesystem(
                gcp_bucket_name, 
                root_path=dl_root_directory, 
                gcp_credentials_path=gcp_credentials_path)
        else:
            raise Exception(f"Invalid fs_type provided, provided {dl_fs_type}")
        
        #Initialize checkpoint filesystem
        ckpt_conf = self.cfg["diffusion_auto_encoder"]["checkpoints"]
        ckpt_fs_type = ckpt_conf["fs_type"]
        ckpt_root_directory = ckpt_conf["data_root_directory"]
        
        if ckpt_fs_type == "local":
            self.ckpt_fs = sdl.os_filesystem(ckpt_root_directory)
        elif ckpt_fs_type == "gcp":
            self.ckpt_fs = sdl.gcp_filesystem(
                gcp_bucket_name, 
                root_path=ckpt_root_directory, 
                gcp_credentials_path=gcp_credentials_path)
        else:
            raise Exception(f"Invalid fs_type provided, provided {ckpt_root_directory}")

    
    def parse_args(self):
        self.credentials_path = self.args.gcs_json
        self.gcs_bucket = self.args.gcs_bucket
        self.ckpt_path = self.args.ckpt_path
        self.log_file = self.args.ckpt_path

    def init_data_loader(self, mode):
        if mode == "train":
            def worker_interface_factory():
                iwi = sdl.ImageWorkerInterface(self.data_fs)
                return iwi
            
            def shard_interface_factory():
                isi = sdl.ImageShardInterface(self.dist_manager)
                return isi
        
        elif mode == "autoencode":
            def worker_interface_factory():
                iwi = sdl.VideoWorkerInterface(self.data_fs)
                return iwi
            
            def shard_interface_factory():
                isi = sdl.VideoShardInterface(self.dist_manager)
                return isi
        
        self.sharded_data_downloader =  sdl.ShardedDataDownloader(
            worker_interface_factory,
            shard_interface_factory,
            self.dist_manager,
        )

    def init_dist_manager(self):
        dm_cfg = self.cfg["dist_manager"]

        mesh_shape = dm_cfg["mesh_shape"]

        self.dist_manager = du.DistManager(mesh_shape, self.credientials_path)
    
    def make_model(self):
        model_conf = self.conf["diffusion_auto_encoder"]["model"]
        enc_conf = model_conf["encoder"]
        dec_conf = model_conf["decoder"]
        
        self.state["prng_key"], enc_key, dec_key = jax.random.split(self.state["prng_key"],3)

        self.state["model"] = DAEModel(
            encoder=daed.Encoder(
                self.dist_manager, 
                key=enc_key, 
                k =enc_conf["k"],
                n_layers=enc_conf["n_layers"]
            ),
            decoder=daed.Decoder(
                self.dist_manager, 
                key=dec_key, 
                k =dec_conf["k"],
                n_layers=dec_conf["n_layers"]
            )
        )
    
    def make_optimizer(self):
        opt_cfg = self.cfg["diffusion_auto_encoder"]["train"]
        
        self.optimizer = optax.adam(lr=opt_cfg["lr"])
        self.state["opt_state"] = self.optimizer.init(self.model)

    def save_checkpoint(self, step):
        """Save a checkpoint at the given step."""
        path = self.new_ckpt_path(step)
        
        # Ensure the checkpoint directory exists
        self.ckpt_fs.makedirs(path, recreate=True)

        # Save model
        model_path = f"{path}/model"
        self.ckpt_fs.makedirs(f"{model_path}/encoder", recreate=True)
        self.ckpt_fs.makedirs(f"{model_path}/decoder", recreate=True)
        self.model.encoder.save(f"{model_path}/encoder")
        self.model.decoder.save(f"{model_path}/decoder")

        # Save optimizer state
        opt_path = f"{path}/opt_state"
        for key, value in self.state["opt_state"].items():
            if hasattr(value, "save"):
                self.ckpt_fs.makedirs(f"{opt_path}/{key}/encoder", recreate=True)
                self.ckpt_fs.makedirs(f"{opt_path}/{key}/decoder", recreate=True)
                value.encoder.save(f"{opt_path}/{key}/encoder")
                value.decoder.save(f"{opt_path}/{key}/decoder")
            else:
                # TODO: Handle other types of optimizer state
                pass
    
    def _list_checkpoints(self):
        """List all checkpoint directories."""
        try:
            directories = [d for d in self.ckpt_fs.listdir('/') if self.ckpt_fs.isdir(d)]
            checkpoint_dirs = [d for d in directories if d.startswith('ckpt_')]
            return sorted(checkpoint_dirs, key=lambda x: int(x.split('_')[1]))
        except fs.errors.ResourceNotFound:
            return []

    
    def load_checkpoint(self, path=None):
        """Load a checkpoint from the given path or the latest if not specified."""
        if path is None:
            path = self.latest_ckpt_path()
        
        if path is None:
            raise ValueError("No checkpoint found to load.")

        # Load model
        model_path = f"{path}/model"
        self.model.encoder.load(f"{model_path}/encoder")
        self.model.decoder.load(f"{model_path}/decoder")

        # Load optimizer state
        opt_path = f"{path}/opt_state"
        for key, value in self.state["opt_state"].items():
            if hasattr(value, "load"):
                value.encoder.load(f"{opt_path}/{key}/encoder")
                value.decoder.load(f"{opt_path}/{key}/decoder")
            else:
                # TODO: Handle other types of optimizer state
                pass
    
    def most_recent_ckpt(self):
        """Get the most recent checkpoint number."""
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return 0
        return int(checkpoints[-1].split('_')[1])


    def new_ckpt_path(self, id):
        ckpt_n = self.most_recent_ckpt()

        ckpt_freq = self.cfg["diffusion_autoencoder"]["train"]["ckpt_freq"]
        
        new_ckpt_n = ckpt_n + ckpt_freq
        assert id == new_ckpt_n

        os.path.join(self.args.ckpt_path,"ckpt_path")

    def latest_ckpt_path(self):
        """Get the path of the latest checkpoint."""
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None
        return f'/{checkpoints[-1]}'
    
    def train(self):
        args = self.args
        cfg = self.cfg

        ckpt_freq = cfg["diffusion_auto_encoder"]["train"]["ckpt_freq"]

        if args.ckpt:
            self.load_checkpoint(args.ckpt)
        else:
            self.load_checkpoint()  # Attempt to load the latest checkpoint

        step = self.most_recent_ckpt()

        for data in self.data_loader:
            step += 1
            loss, self.state = dc.update(self.state, data, self.optimizer, self.loss_fn)
            print(f"Step {step}, Loss: {loss}")

            if step % ckpt_freq == 0:
                self.save_checkpoint(step)


    def autoencode(self):
        args = self.args
        cfg = self.cfg

        latest_ckpt_path = self.latest_ckpt_path()
        self.load_checkpoint(latest_ckpt_path)

    



