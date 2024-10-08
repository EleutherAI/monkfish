"""
diffusion_ar.py

This module implements a Diffusion Autoregressive (AR) model for video generation using JAX and other related libraries.

The DiffARHarness class serves as the main interface for training and sampling from the model. It handles:
1. Initialization of file systems for data loading and checkpointing
2. Distributed training setup
3. Model creation and optimization
4. Data loading and preprocessing
5. Training loop implementation with checkpointing
6. Sampling from the trained model

Key features:
- Supports both local and Google Cloud Platform (GCP) file systems
- Implements a sharded data loader for efficient distributed training
- Uses a transformer-based architecture for the diffusion model
- Provides checkpoint management for model saving and loading
- Implements a sampling procedure for generating video sequences

The module is designed to be flexible and configurable, with most parameters
specified in a configuration dictionary passed to the DiffARHarness constructor.

Usage:
- For training: Initialize DiffARHarness with appropriate arguments and call the train() method
- For sampling: Initialize DiffARHarness, load a checkpoint, and call the sample() method

Note: This implementation is part of a larger project and may require additional
modules and configurations to run successfully.
"""

import os
import time
import re
import collections
import functools

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
                credentials_path=gcp_credentials_path)
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
        elif operation == "sample":
            pass
        else:
            raise ValueError(f"Unsupported operation {operation}")

        
        if operation in ["train_adm"]:
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

        self.state["model"] = dard.SplitTransformerARDM(
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

        opt_state = self.optimizer.init(self.state["model"])

        #Counts are on device 0 by default, we scatter to prevent save/load inconsistencies 
        scattered_count = self.dist_manager.scatter(
            self.dist_manager.uniform_sharding, jnp.int32)(opt_state[0].count)
        self.state["opt_state"] = (opt_state[0]._replace(count=scattered_count),) + opt_state[1:]
    
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
            txt, true_x = data
            diff_data = ((txt, true_x), true_x)
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
            self.load_checkpoint(ckpt_path)
        else:
            #Start from scratch and use existing init without loading a checkpoint
            step = 0

        # Initialize the data downloader
        self.sharded_data_downloader.start(step * self.sharded_data_downloader.batch_size)


        with jax.default_matmul_precision('bfloat16'):
            try:
                total_loss = 0
                log_start_step = step
                step_time = time.time()
                while step < total_steps:
                    #Save checkpoint 
                    if step % ckpt_freq == 0:
                        print(f"Saving checkpoint for step {step}...")
                        ckpt_path = self.ckpt_path(step)
                        self.save_checkpoint(ckpt_path)
                    
                    step += 1
                    
                    # Get data from the downloader
                    t1 = time.time()
                    data = self.sharded_data_downloader.step()
                    t2 = time.time()

                    # Update the model
                    loss, self.state = dc.update_state_dict(self.state, data, self.optimizer, loss_fn)
                    t3 = time.time()
                    if step % log_freq == 0:
                        new_time = time.time() 
                        it_per_s = log_freq/(new_time-step_time)
                        #print((t2-t1)/(t3-t1))
                        #print(t2-t1)
                        #print(t3-t1)

                        log_object = {
                            "step": step,
                            "loss": round(loss, 3),
                            "iterations_per_second": round(it_per_s, 1),
                            "data_loader_time_fraction": round((t2-t1)/(t3-t1), 3)
                        }

                        print(log_object)
                        
                        #yield log_object

                        step_time = new_time 
                    
                    # Acknowledge that we've processed the data
                    self.sharded_data_downloader.ack()

            except KeyboardInterrupt:
                print("Training interrupted.")
            finally:
                # Always stop the data downloader when we're done
                self.sharded_data_downloader.stop()
                print("Data downloader stopped.")

        print("Training completed.")

        model_conf = self.cfg["transformer_ardm"]["model"]
        model_conf["io_dim"]
    
    #TODO: Make architecture independent of noise level and implement efficient sampling
    def sample(self, token_prompt, prefix_prompt):
        args = self.args
        cfg = self.cfg

        bs = 1

        # Load the latest checkpoint
        latest_ckpt_path = self.latest_ckpt_path()
        self.load_checkpoint(latest_ckpt_path)

        # Get sampling configuration
        sample_cfg = cfg["transformer_ardm"]["sample"]
        model_cfg = self.cfg["transformer_ardm"]["model"]
        model_cfg["io_dim"]
        n_steps = sample_cfg["n_steps"]  # Number of diffusion steps
        latent_length = sample_cfg["latent_length"]  # Number of latents to generate

        model = self.state["model"]
        model_f = functools.partial(model, mode="generate")
        seed = sample_cfg["seed"]
        
        # Load prompts across cluster
        if token_prompt is not None:
            input_tok_local = jnp.repeat(jnp.array(token_prompt)[None, :], bs, axis=0)
            # Assert that the first token is bos_token
            assert input_tok_local[0, 0] == sample_cfg["bos_token"], "The first token must be the bos_token"
        else:
            input_tok_local = jnp.array([[sample_cfg["bos_token"]]] * bs)

        if prefix_prompt is not None:
            input_lat_local = jnp.repeat(jnp.array(prefix_prompt)[None, :, :], bs, axis=0)
        else:
            input_lat_local = jnp.zeros((bs, 0, model_cfg["io_dim"]))

        # Create distributed arrays
        input_tok = jax.make_array_from_process_local_data(self.dist_manager.uniform_sharding, input_tok_local)
        input_lat = jax.make_array_from_process_local_data(self.dist_manager.uniform_sharding, input_lat_local)

        # Initialize output sequence
        key = self.dist_manager.get_key(seed)
        output_sequence = jnp.zeros((bs, 0, model_cfg["io_dim"]))

        for _ in range(latent_length):
            
            key, subkey = jax.random.split(key)
            
            # Get the shape for the next token
            next_token_shape = (input_lat.shape[1]+1, model_cfg["io_dim"])
            
            input_data = (input_tok, input_lat)
            
            # Sample the next token using diffusion
            vector_output = dc.sample_diffusion(
                inputs=input_data,
                model=model_f,
                f_neg_gamma=dc.f_neg_gamma,
                key=subkey,
                n_steps=n_steps,
                shape=next_token_shape
            )
            next_vector = vector_output[:,-1]
            
            # Append the new vector to the output sequence
            output_sequence = jnp.concatenate([output_sequence, next_vector[jnp.newaxis, :]], axis=1)
            
            # Update input sequence for next iteration
            input_lat = jnp.concatenate([input_lat, next_vector[jnp.newaxis, :]], axis=1)
            
            # Check for end of sequence condition
            if self.is_end_of_sequence(output_sequence, latent_length):
                break
            print("-----------")

        # Only return the output sequence if this is the process with rank 0
        if self.dist_manager.pid == 0:
            return output_sequence
        else:
            return None

    def is_end_of_sequence(self, sequence, video_length):
        # Check if the sequence has reached the desired video length
        return sequence.shape[0] >= video_length

    def save_samples(self, generated_sequences):
        # Implement saving or processing the generated samples
        print("Generated samples shape:", generated_sequences.shape)
        # TODO: Implement saving the generated video to a file or further processing
