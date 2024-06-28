import os
import re
import collections

import fs
import jax
import jax.numpy as jnp
import optax
import pickle as pkl

import catfish.lvd.models.dist_autoencoding_diffusion as daed
import catfish.lvd.models.dist_utils as du
import catfish.lvd.shrd_data_loader as sdl
import catfish.lvd.diffusion_core as dc


DAEModel = collections.namedtuple(["encoder", "decoder"])

class DiffARHarness:
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

    def init_data_loader(self, mode):
        if mode == "train":
            def worker_interface_factory():
                lwi = sdl.LatentWorkerInterface(self.data_fs)
                return lwi
            
            def shard_interface_factory():
                lwi = sdl.ImageShardInterface(self.dist_manager)
                return lwi
        
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
        model_conf = self.conf["transformer_ardm"]["model"]
        
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
    
    def _list_checkpoints(self):
        """List all checkpoint directories."""
        try:
            directories = [d for d in self.ckpt_fs.listdir('/') if self.ckpt_fs.isdir(d)]
            checkpoint_dirs = [d for d in directories if d.startswith('ckpt_')]
            return sorted(checkpoint_dirs, key=lambda x: int(x.split('_')[1]))
        except fs.errors.ResourceNotFound:
            return []

    def save_checkpoint(self, step):
        """Save a checkpoint at the given step."""
        path = self.new_ckpt_path(step)
        
        # Ensure the checkpoint directory exists
        self.ckpt_fs.makedirs(path, recreate=True)

        # Save model
        model_path = f"{path}/model"
        self.ckpt_fs.makedirs(f"{model_path}/encoder", recreate=True)
        self.ckpt_fs.makedirs(f"{model_path}/decoder", recreate=True)
        self.state["model"].encoder.save(f"{model_path}/encoder")
        self.state["model"].decoder.save(f"{model_path}/decoder")

        # Save optimizer state
        opt_path = f"{path}/opt_state"
        self.ckpt_fs.makedirs(opt_path, recreate=True)

        def save_opt_state(opt_state, prefix):
            for key, value in opt_state.items():
                if isinstance(value, (jnp.ndarray, jax.Array)):
                    self.dist_manager.save_array(value, self.dist_manager.uniform_sharding, f"{prefix}/{key}")
                elif isinstance(value, (dict, optax.EmptyState)):
                    save_opt_state(value, f"{prefix}/{key}")
                else:
                    # For scalar values or other types, save using regular pickle
                    with self.ckpt_fs.open(f"{prefix}/{key}.pkl", 'wb') as f:
                        pkl.dump(value, f)

        save_opt_state(self.state["opt_state"], opt_path)

        # Save PRNG key
        self.dist_manager.save_array(self.state["prng_key"], self.dist_manager.uniform_sharding, f"{path}/prng_key")

    def load_checkpoint(self, path=None):
        """Load a checkpoint from the given path or the latest if not specified."""
        if path is None:
            path = self.latest_ckpt_path()
        
        if path is None:
            raise ValueError("No checkpoint found to load.")

        # Load model
        model_path = f"{path}/model"
        self.state["model"].encoder.load(f"{model_path}/encoder")
        self.state["model"].decoder.load(f"{model_path}/decoder")

        # Load optimizer state
        opt_path = f"{path}/opt_state"

        def load_opt_state(prefix):
            opt_state = {}
            for item in self.ckpt_fs.listdir(prefix):
                item_path = f"{prefix}/{item}"
                if self.ckpt_fs.isdir(item_path):
                    opt_state[item] = load_opt_state(item_path)
                elif item.endswith('.pkl'):
                    with self.ckpt_fs.open(item_path, 'rb') as f:
                        opt_state[item[:-4]] = pkl.load(f)
                else:
                    opt_state[item] = self.dist_manager.load_array(self.dist_manager.uniform_sharding, item_path)
            return opt_state

        self.state["opt_state"] = load_opt_state(opt_path)

        # Reconstruct the OptState object
        self.state["opt_state"] = jtu.tree_map(
            lambda x: x if isinstance(x, (optax.EmptyState, dict)) else x,
            self.state["opt_state"]
        )

        # Load PRNG key
        self.state["prng_key"] = self.dist_manager.load_array(self.dist_manager.uniform_sharding, f"{path}/prng_key")

    
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

        train_cfg = cfg["diffusion_auto_encoder"]["train"]
        ckpt_freq = train_cfg["ckpt_freq"]
        total_steps = train_cfg["total_steps"]
        log_freq = train_cfg["log_freq"]

        if args.ckpt:
            self.load_checkpoint(args.ckpt)
        else:
            self.load_checkpoint()  # Attempt to load the latest checkpoint

        step = self.most_recent_ckpt()

        # Initialize the data downloader
        self.sharded_data_downloader.start(step * self.sharded_data_downloader.batch_size)

        try:
            total_loss = 0
            log_start_step = step
            while step < total_steps:
                step += 1
                
                # Get data from the downloader
                data = self.sharded_data_downloader.step()

                # Update the model
                loss, self.state = dc.update(self.state, data, self.optimizer, self.loss_fn)

                # Accumulate loss
                total_loss += loss

                # Acknowledge that we've processed the data
                self.sharded_data_downloader.ack()

                if step % log_freq == 0:
                    avg_loss = total_loss / (step - log_start_step)
                    print(f"Step {step}, Average Loss: {avg_loss:.4f}")
                    # Reset for next logging interval
                    total_loss = 0
                    log_start_step = step

                if step % ckpt_freq == 0:
                    self.save_checkpoint(step)

        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint...")
            self.save_checkpoint(step)
        finally:
            # Always stop the data downloader when we're done
            self.sharded_data_downloader.stop()

        print("Training completed.")

    def autoencode(self):
        args = self.args
        cfg = self.cfg

        latest_ckpt_path = self.latest_ckpt_path()
        self.load_checkpoint(latest_ckpt_path)

    



