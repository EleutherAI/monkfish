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

import monkfish.lvd.models.dist_autoreg_diffusion as dard
import monkfish.lvd.models.dist_utils as du
import monkfish.lvd.shrd_data_loader as sdl
import monkfish.lvd.diffusion_core as dc
import monkfish.lvd.fs_utils as fs_utils

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
        dl_conf = self.cfg["transformer_ardm"]["data_loader"]
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
        ckpt_conf = self.cfg["transformer_ardm"]["checkpoints"]
        ckpt_fs_type = ckpt_conf["fs_type"]
        ckpt_root_directory = ckpt_conf["ckpt_root_directory"]
        
        if ckpt_fs_type == "local":
            self.ckpt_fs = fs_utils.os_filesystem(ckpt_root_directory)
        elif ckpt_fs_type == "gcp":
            self.ckpt_fs = fs_utils.gcp_filesystem(
                gcp_bucket_name, 
                root_path=ckpt_root_directory, 
                gcp_credentials_path=gcp_credentials_path)
        else:
            raise Exception(f"Invalid fs_type provided, provided {ckpt_root_directory}")

    def init_data_loader(self):
        operation = self.args.operation
        dl_conf = self.cfg["transformer_ardm"]["data_loader"]

        if operation == "train_adm":
            worker_interface_cls = sdl.LatentWorkerInterface
            
            def shard_interface_factory():
                isi = sdl.LatentShardInterface(self.dist_manager)
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
        dm_cfg = self.cfg["transformer_ardm"]["dist_manager"]

        mesh_shape = dm_cfg["mesh_shape"]

        self.dist_manager = du.DistManager(mesh_shape, self.ckpt_fs)
    
    def make_model(self):
        model_conf = self.cfg["transformer_ardm"]["model"]

        seed = self.cfg["seed"]
        self.state["prng_key"] = self.dist_manager.get_key(seed)
        
        self.state["prng_key"], model_key = jax.random.split(self.state["prng_key"], 2)

        self.state["model"] = dard.TransformerARDM(
            self.dist_manager,
            key=model_key,
            res_dim=model_conf["res_dim"],
            io_dim=model_conf["io_dim"],
            vocab=model_conf["vocab"],
            n_layers=model_conf["n_layers"],
            mlp_dim=model_conf["mlp_dim"],
            qk_dim=model_conf["qk_dim"],
            v_dim=model_conf["v_dim"],
            n_head=model_conf["n_head"],
        )
    
    def make_optimizer(self):
        opt_cfg = self.cfg["transformer_ardm"]["train"]
        
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

        ckpt_freq = self.cfg["transformer_ardm"]["train"]["ckpt_freq"]
        
        new_ckpt_n = ckpt_n + ckpt_freq

        new_ckpt_path = f"/ckpt_{new_ckpt_n}"
        return new_ckpt_path
    

    def train(self):
        def loss_fn(model, data, subkey):
            diff_data = (data, data[1])
            loss = dc.diffusion_loss(
                model, diff_data, dc.f_neg_gamma, subkey)
            return loss
        
        args = self.args
        cfg = self.cfg

        train_cfg = cfg["transformer_ardm"]["train"]
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
    
    def sample(self):
        args = self.args
        cfg = self.cfg

        # Load the latest checkpoint
        latest_ckpt_path = self.latest_ckpt_path()
        self.load_checkpoint(latest_ckpt_path)

        # Get sampling configuration
        sample_cfg = cfg["transformer_ardm"]["sample"]
        n_steps = sample_cfg.get("n_steps", 100)  # Number of diffusion steps
        video_length = sample_cfg.get("video_length", 100)  # Number of frames to generate
        batch_size = sample_cfg.get("batch_size", 1)  # Number of samples to generate
        prompt = sample_cfg.get("prompt", None)  # Initial prompt, if any

        model = self.state["model"]
        key = self.state["prng_key"]

        # Initialize input sequence with prompt or empty
        if prompt is not None:
            input_sequence = jnp.array([prompt] * batch_size)
        else:
            input_sequence = jnp.zeros((batch_size, 0, model.io_dim))

        # Initialize output sequence
        output_sequence = jnp.zeros((batch_size, 0, model.io_dim))

        for _ in range(video_length):
            key, subkey = jax.random.split(key)
            
            # Get the shape for the next token
            next_token_shape = (model.io_dim,)
            
            # Sample the next token using diffusion
            next_token = dc.sample_diffusion(
                inputs=input_sequence,
                model=model,
                f_neg_gamma=dc.f_neg_gamma,
                key=subkey,
                n_steps=n_steps,
                shape=next_token_shape
            )
            
            # Append the new token to the output sequence
            output_sequence = jnp.concatenate([output_sequence, next_token[:, jnp.newaxis, :]], axis=1)
            
            # Update input sequence for next iteration
            input_sequence = jnp.concatenate([input_sequence, next_token[:, jnp.newaxis, :]], axis=1)
            
            # Check for end of sequence condition
            if self.is_end_of_sequence(output_sequence, video_length):
                break

        # Save or process the generated samples
        self.save_samples(output_sequence)

        return output_sequence

    def is_end_of_sequence(self, sequence, video_length):
        # Check if the sequence has reached the desired video length
        return sequence.shape[1] >= video_length

    def save_samples(self, generated_sequences):
        # Implement saving or processing the generated samples
        print("Generated samples shape:", generated_sequences.shape)
        # TODO: Implement saving the generated video to a file or further processing
