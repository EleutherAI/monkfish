import os
import time
import re
import collections

import fs
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import optax
import pickle as pkl

import catfish.lvd.models.dist_autoencoding_diffusion as daed
import catfish.lvd.models.dist_utils as du
import catfish.lvd.shrd_data_loader as sdl
import catfish.lvd.diffusion_core as dc


DAEModel = collections.namedtuple('DAEModel', ['encoder', 'decoder'])

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
        self.worker_fs_args = None
        self.ckpt_fs = None

        print("Parsing arguments...")
        self.parse_args()
        
        print("Initializing file system...")
        self.init_fs()

        print("Initializing sharding manager...")
        self.init_dist_manager()
        
        print("Initializing data_loader...")
        self.init_data_loader()
        
        print("Creating model...")
        self.make_model()

        print("Creating Optimizer...")
        self.make_optimizer()

    def parse_args(self):
        pass

    def init_fs(self):
        gcp_conf = self.cfg["gcp"]
        gcp_credentials_path =  gcp_conf["gcp_credentials_path"]
        gcp_bucket_name =  gcp_conf["gcp_bucket_name"]

        #Initialize data loader file system
        dl_conf = self.cfg["diffusion_auto_encoder"]["data_loader"]
        dl_fs_type = dl_conf["fs_type"]
        dl_root_directory = dl_conf["data_root_directory"]

        if dl_fs_type == "local":
            self.worker_fs_args = {
                "fs_type": "os",
                "root_path": dl_root_directory
            }
        elif dl_fs_type == "gcp":
            self.worker_fs_args = {
                "fs_type": "gcp",
                "bucket_name": gcp_bucket_name,
                "root_path": dl_root_directory,
                "credentials_path": gcp_credentials_path
            }
        else:
            raise Exception(f"Invalid fs_type provided, provided {dl_fs_type}")
        
        #Initialize checkpoint filesystem
        ckpt_conf = self.cfg["diffusion_auto_encoder"]["checkpoints"]
        ckpt_fs_type = ckpt_conf["fs_type"]
        ckpt_root_directory = ckpt_conf["ckpt_root_directory"]
        
        if ckpt_fs_type == "local":
            self.ckpt_fs = sdl.os_filesystem(ckpt_root_directory)
        elif ckpt_fs_type == "gcp":
            self.ckpt_fs = sdl.gcp_filesystem(
                gcp_bucket_name, 
                root_path=ckpt_root_directory, 
                gcp_credentials_path=gcp_credentials_path)
        else:
            raise Exception(f"Invalid fs_type provided, provided {ckpt_root_directory}")

    def init_data_loader(self):
        operation = self.args.operation
        dl_conf = self.cfg["diffusion_auto_encoder"]["data_loader"]

        if operation == "train_dae":
            worker_interface_cls = sdl.ImageWorkerInterface
            
            def shard_interface_factory():
                isi = sdl.ImageShardInterface(self.dist_manager)
                return isi
        
        elif operation == "autoencode":
            worker_interface_cls = sdl.VideoWorkerInterface
            
            def shard_interface_factory():
                isi = sdl.VideoShardInterface(self.dist_manager)
                return isi
        else:
            raise ValueError(f"Unsupported operation {operation}")

        
        self.sharded_data_downloader =  sdl.ShardedDataDownloader(
            self.worker_fs_args,
            worker_interface_cls,
            shard_interface_factory,
            self.dist_manager,
            workers_per_node=dl_conf["workers_per_node"],
            batch_size=dl_conf["batch_size"],
            queue_depth=dl_conf["queue_depth"],
        )

    def init_dist_manager(self):
        dm_cfg = self.cfg["diffusion_auto_encoder"]["dist_manager"]

        mesh_shape = dm_cfg["mesh_shape"]

        self.dist_manager = du.DistManager(mesh_shape, self.ckpt_fs)
    
    def make_model(self):
        model_conf = self.cfg["diffusion_auto_encoder"]["model"]
        enc_conf = model_conf["encoder"]
        dec_conf = model_conf["decoder"]

        seed = self.cfg["seed"]
        self.state["prng_key"] = self.dist_manager.get_key(seed)
        
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
        
        self.optimizer = optax.adam(learning_rate=opt_cfg["lr"])
        self.state["opt_state"] = self.optimizer.init(self.state["model"])
    
    def list_checkpoints(self):
        """List all checkpoint directories."""
        try:
            directories = [d for d in self.ckpt_fs.listdir('/') if self.ckpt_fs.isdir(d)]
            checkpoint_dirs = [f"/{d}" for d in directories if d.startswith('ckpt_')]
            return sorted(checkpoint_dirs, key=lambda x: int(x.split('_')[1]))
        except fs.errors.ResourceNotFound:
            return []

    def save_checkpoint(self, path):
        """Save a checkpoint at the given step."""
        
        ckpt_file_path = f"{path}/ckpt.pkl"
        sharding_pytree = self.dist_manager.get_pytree_sharding(self.state)
        self.dist_manager.save_pytree(self.state, sharding_pytree, ckpt_file_path)

    def load_checkpoint(self, path):
        """Load a checkpoint from the given path"""
        ckpt_file_path = f"{path}/ckpt.pkl"
        sharding_pytree = self.dist_manager.get_pytree_sharding(self.state)
        self.state = self.dist_manager.load_pytree(sharding_pytree, ckpt_file_path)
    
    def latest_ckpt_step(self):
        """Get the most recent checkpoint number."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None 
        return int(checkpoints[-1].split('_')[1])

    def latest_ckpt_path(self):
        """Get the path of the latest checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]
    
    def ckpt_path(self, step):
        return f"/ckpt_{step}"

    def new_ckpt_path(self):

        ckpt_n = self.most_recent_ckpt_step()

        ckpt_freq = self.cfg["diffusion_auto_encoder"]["train"]["ckpt_freq"]
        
        new_ckpt_n = ckpt_n + ckpt_freq

        new_ckpt_path = f"/ckpt_{new_ckpt_n}"
        return new_ckpt_path
    

    def train(self):
        def loss_fn(model, data, subkey):
            latents = jax.vmap(model.encoder)(data)
            diff_data = (latents, data)
            loss = dc.diffusion_loss(
                model.decoder, diff_data, dc.f_neg_gamma, subkey)
            return loss
        
        args = self.args
        cfg = self.cfg

        train_cfg = cfg["diffusion_auto_encoder"]["train"]
        ckpt_freq = train_cfg["ckpt_freq"]
        total_steps = train_cfg["total_steps"]
        log_freq = train_cfg["log_freq"]

        if args.ckpt:
            #Load specified checkpoint
            step = args.ckpt
            ckpt_path = self.ckpt_path(step)
            self.load_checkpoint(ckpt_path)
        elif self.list_checkpoints():
            #Load latest checkpoint
            step = self.latest_ckpt_step()
            ckpt_path = self.latest_ckpt_path()
            #self.load_checkpoint(ckpt_path)
        else:
            #Start from scratch and use existing init without loading a checkpoint
            step = 0

        # Initialize the data downloader
        self.sharded_data_downloader.start(step * self.sharded_data_downloader.batch_size)

        try:
            total_loss = 0
            log_start_step = step
            while step < total_steps:
                
                #Save checkpoint 
                if step % ckpt_freq == 0:
                    print(f"Saving checkpoint for step {step}...")
                    ckpt_path = self.ckpt_path(step)
                    self.save_checkpoint(ckpt_path)
                
                step += 1

                if step > 35:
                    break

                time.sleep(1)
                
                # Get data from the downloader
                data = self.sharded_data_downloader.step()
                print("a")

                # Update the model
                loss, self.state = dc.update_state_dict(self.state, data, self.optimizer, loss_fn)
                print("b")
                print(loss)

                """
                # Accumulate loss
                total_loss += loss


                if step % log_freq == 0:
                    avg_loss = total_loss / (step - log_start_step)
                    print(f"Step {step}, Average Loss: {avg_loss:.4f}")
                    # Reset for next logging interval
                    total_loss = 0
                    log_start_step = step
                """
                
                # Acknowledge that we've processed the data
                self.sharded_data_downloader.ack()
                print("c")

        except KeyboardInterrupt:
            print("Training interrupted.")
        finally:
            # Always stop the data downloader when we're done
            print("d")
            self.sharded_data_downloader.stop()
            print("e")

        print("Training completed.")

    def autoencode(self):
        args = self.args
        cfg = self.cfg

        latest_ckpt_path = self.latest_ckpt_path()
        self.load_checkpoint(latest_ckpt_path)