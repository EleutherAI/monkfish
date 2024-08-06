import fs
import os
import time

import argparse
import json
import multiprocessing


import monkfish.lvd.diffusion_ae as dae
import monkfish.lvd.diffusion_ar as dar
import monkfish.lvd.fs_utils as fs_utils

import monkfish.tpu as tpu
import monkfish.tpu.infrastructure as tpu_infra

def configure_globals():
    multiprocessing.set_start_method('spawn')

def parse_args():
    parser = argparse.ArgumentParser(description="Sharded/distributed training tool for video generative model")

    # Required arguments
    parser.add_argument("config", help="Path to configuration file (JSON)")
    parser.add_argument("mode", choices=["local", "distributed", "swarm"], help="Mode of operation: local (single node), distributed (multinode), or swarm")

    # Subparsers for different operations
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Operation to perform")

    # Training diffusion autoencoder
    train_dae_parser = subparsers.add_parser("train_dae", help="Train the diffusion autoencoder")
    train_dae_parser.add_argument("--ckpt", default=None, help="Path to checkpoint to resume training from")
    
    # Lifting videos
    lift_parser = subparsers.add_parser("lift", help="Lift videos into the diffusion latent space")
    lift_parser.add_argument("input_videos", nargs="+", help="Input video files")

    # Training autoregressive diffusion model
    train_adm_parser = subparsers.add_parser("train_adm", help="Train the autoregressive diffusion model")
    train_adm_parser.add_argument("--ckpt", default=None, help="Path to checkpoint to resume training from")

    # Reconstructing static test image
    reconstruct_parser = subparsers.add_parser("reconstruct", help="Reconstruct a static test image")
    reconstruct_parser.add_argument("input_image", help="Input image file")

    # Sampling
    sample_parser = subparsers.add_parser("sample", help="Sample a video using both models")
    sample_parser.add_argument("--text_prompt", help="Text prompt for sampling")
    sample_parser.add_argument("--image_prompt", help="Image prompt for sampling")
    sample_parser.add_argument("--video_prompt", help="Video prompt for sampling")

    # Clean dir 
    clear_dir_parser = subparsers.add_parser("clear_fs", help="Deletes specified directories")
    clear_dir_parser.add_argument("--fs_type", type=str, help="file system type")
    clear_dir_parser.add_argument("--root_dir", type=str, help="root directory")
    clear_dir_parser.add_argument("--target_dirs", nargs="*", help="list of target directories to clear and remove")

    
    # List dir 
    list_fs_parser = subparsers.add_parser("list_fs", help="List a specified directory")
    list_fs_parser.add_argument("--fs_type", type=str, help="file system type")
    list_fs_parser.add_argument("--root_dir", type=str, help="root_dir")
    list_fs_parser.add_argument("--recursive", action="store_true", help="List directories recursively")

    return parser.parse_args()

def read_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    configure_globals()
    args = parse_args()
    config = read_config(args.config)

    # Process the arguments and call appropriate functions
    if args.operation == "train_dae":
        train_diffusion_autoencoder(config, args)
    elif args.operation == "lift":
        lift_videos(config, args)
    elif args.operation == "train_adm":
        print("A")
        train_autoregressive_diffusion_model(config, args)
        print("C")
    elif args.operation == "reconstruct":
        reconstruct_image(config, args)
    elif args.operation == "sample":
        sample_video(config, args)
    elif args.operation == "clear_fs":
        clear_filesystem(config, args)
    elif args.operation == "list_fs":
        list_filesystem(config, args)

def setup_filesystem(config, args):
    fs_type = args.fs_type
    root_dir = args.root_dir

    fs_args = {
        'fs_type': fs_type,
        'root_path': root_dir,
        'bucket_name': config['gcp']['gcp_bucket_name'],
        'credentials_path': config['gcp']['gcp_credentials_path']
    }

    try:
        filesystem = fs_utils.fs_initializer(fs_args)
        return filesystem, fs_type, root_dir
    except Exception as e:
        print(f"Error setting up {fs_type} filesystem: {e}")
        return None, fs_type, root_dir

def clear_filesystem(config, args):
    filesystem, fs_type, root_dir = setup_filesystem(config, args)
    if not filesystem:
        return

    # New argument for target directories
    target_dirs = args.target_dirs if hasattr(args, 'target_dirs') else []

    try:
        if filesystem.exists('/'):
            if not target_dirs:
                # If no specific targets, clear everything
                fs_utils.clear_and_remove_dir(filesystem, '/')
            else:
                for target in target_dirs:
                    # Use '/' as the base, since the filesystem is already rooted at the specified root_dir
                    full_path = fs.path.combine('/', target)
                    if filesystem.exists(full_path):
                        fs_utils.clear_and_remove_dir(filesystem, full_path)
                        print(f"Cleared and removed directory: {full_path}")
                    else:
                        print(f"Directory does not exist: {full_path}")
        else:
            print(f"Root directory does not exist: {root_dir}")
    except Exception as e:
        print(f"Error clearing {fs_type} filesystem: {e}")
    finally:
        filesystem.close()

def list_filesystem(config, args):
    filesystem, fs_type, root_dir = setup_filesystem(config, args)
    if not filesystem:
        return
    
    print(f"Listing contents of {fs_type} filesystem at {root_dir}")

    def print_dir(path, level=0, recursive=False):
        indent = ' ' * 4 * level
        print(f"{indent}{fs.path.basename(path) or path}/")
        try:
            for item in filesystem.scandir(path):
                if item.is_dir:
                    if recursive:
                        try:
                            print_dir(f"{path}/{item.name}", level + 1, recursive)
                        except fs.errors.ResourceNotFound:
                            print(f"{indent}    {item.name}/ (inaccessible)")
                    else:
                        print(f"{indent}    {item.name}/")
                else:
                    print(f"{indent}    {item.name}")
        except PermissionError:
            print(f"{indent}    (Permission denied)")

    try:
        if filesystem.exists('/'):
            print_dir('/', recursive=args.recursive)
        else:
            print(f"Directory does not exist: {root_dir}")
    except Exception as e:
        print(f"Error listing {fs_type} directory: {e}")

def train_diffusion_autoencoder(config, args):
    print(f"Training diffusion autoencoder with config {config} in {args.mode} mode")

    def dae_harness_factory():
        return dae.DiffAEHarness(
            args,
            config
        )

    backend = config["backend"]
    if args.mode == "local":
        if backend == "tpu":
            dae_harness = dae_harness_factory()
            for log_object in dae_harness.train():
                # Assuming log_object is a dictionary with keys: step, loss, iterations_per_second, data_loader_time_fraction
                print(f"Step {log_object['step']}: "
                      f"Loss: {log_object['loss']:.3f}, "
                      f"It/s: {log_object['iterations_per_second']:.1f}, "
                      f"Data Loader Time Frac: {log_object['data_loader_time_fraction']:.3f}")
                
        elif backend == "gpu":
            # TODO: Implement local GPU training
            raise NotImplementedError()
        elif backend == "cpu":
            # TODO: Implement local CPU training
            raise NotImplementedError()
        else:
            print("Invalid backend")

    elif args.mode == "distributed":
        if backend == "tpu":
            ssh_key_path = os.path.expanduser(config["tpu"]["ssh_key_path"])
            tpu_name = config["tpu"]["tpu_name"]
            tpu_size = config["tpu"]["size"]
            region = config["tpu"]["region"]
            preemptible = config["tpu"]["preemptible"]

            head_info, address = tpu_infra.init()
            print(f"TPU head info: {head_info}")

            tpu_cluster = tpu_infra.TPUCluster(tpu_name, tpu_size, region, 
                preemptible, ssh_key_path, head_info, address, owner=False)
            
            try:
                n = config["tpu"]["num_cores"]  # Number of TPU cores to use
                
                # Put the factory function on the TPU cluster
                factory_on_tpu = tpu_cluster.put([dae_harness_factory] * n)
                
                # Call the factory function on each TPU host to create the harnesses
                dae_harnesses = factory_on_tpu(args, config)
                
                # Put the train method of each harness on the TPU
                train_func = dae_harnesses.train
                
                steps = config["diffusion_auto_encoder"]["train"]["total_steps"]
                t1 = time.time()
                for i in range(steps):
                    try:
                        r = None
                        r = train_func()
                        tpu_cluster.get(r)
                        
                        if i % config["diffusion_auto_encoder"]["train"]["ckpt_freq"] == 0:
                            # Save checkpoint logic here
                            pass
                        
                    except tpu_infra.DeadTPUException as e:
                        print("TPU failure detected, restarting...")
                        del train_func
                        del r
                        tpu_cluster.restart()
                        
                        # Recreate the harnesses after restart
                        factory_on_tpu = tpu_cluster.put([dae_harness_factory] * n)
                        dae_harnesses = factory_on_tpu(args, config)
                        train_func = dae_harnesses.train
                
                t2 = time.time()
                print(f"Training completed. Total time: {t2-t1}, Time per step: {(t2-t1)/steps}")
                
            finally:
                tpu_infra.shutdown()
        elif backend == "gpu":
            # TODO: Implement distributed GPU training
            pass
        else:
            # TODO: Implement distributed CPU training
            pass
    elif args.mode == "swarm":
        # TODO: Implement swarm training
        pass
    else:
        print(f"Mode {args.mode} and backend {backend} is not supported for train_dae")

def lift_videos(config, args):
    print(f"Lifting videos {args.input_videos} with config {config} in {args.mode} mode")

    backend = config.get("backend", "cpu")
    if args.mode == "local":
        # TODO: Implement local video lifting
        pass
    elif args.mode == "distributed":
        # TODO: Implement distributed video lifting
        pass
    elif args.mode == "swarm":
        # TODO: Implement swarm video lifting
        pass
    else:
        print(f"Mode {args.mode} is not supported for lift_videos")

def train_autoregressive_diffusion_model(config, args):
    print(f"Training autoregressive diffusion model with config {config} in {args.mode} mode")

    def ardm_harness_factory(args, config):
        return dar.DiffARHarness(
            args,
            config
        )

    backend = config["backend"]
    if args.mode == "local":
        if backend == "tpu":
            ardm_harness = ardm_harness_factory(args, config)
            for log_object in ardm_harness.train():
                print(f"Step {log_object['step']}: "
                      f"Loss: {log_object['loss']:.3f}, "
                      f"It/s: {log_object['iterations_per_second']:.1f}, "
                      f"Data Loader Time Frac: {log_object['data_loader_time_fraction']:.3f}")
                
        elif backend == "gpu":
            # TODO: Implement local GPU training
            raise NotImplementedError("Local GPU training not implemented for ARDM")
        elif backend == "cpu":
            # TODO: Implement local CPU training
            raise NotImplementedError("Local CPU training not implemented for ARDM")
        else:
            print("Invalid backend")

    elif args.mode == "distributed":
        if backend == "tpu":
            # Implement distributed TPU training
            ssh_key_path = os.path.expanduser(config["tpu"]["ssh_key_path"])
            tpu_name = config["tpu"]["tpu_name"]
            tpu_size = config["tpu"]["size"]
            region = config["tpu"]["region"]
            preemptible = config["tpu"]["preemptible"]

            head_info, address = tpu_infra.init()
            print(f"TPU head info: {head_info}")

            tpu_cluster = tpu_infra.TPUCluster(tpu_name, tpu_size, region, 
                preemptible, ssh_key_path, head_info, address, owner=False)
            
            try:
                #n = config["tpu"]["num_cores"]  # Number of TPU cores to use
                n = 4 
                
                # Put the factory function on the TPU cluster
                factory_on_tpu = tpu_cluster.put([ardm_harness_factory] * n)
                args_on_tpu = tpu_cluster.put([args] * n)
                config_on_tpu = tpu_cluster.put([config] * n)
                
                # Call the factory function on each TPU host to create the harnesses
                ardm_harnesses = factory_on_tpu(args_on_tpu, config_on_tpu)
                
                # Put the train method of each harness on the TPU
                train_func = ardm_harnesses.train
                
                steps = config["autoregressive_diffusion_model"]["train"]["total_steps"]
                t1 = time.time()
                for i in range(steps):
                    try:
                        r = None
                        r = train_func()
                        log_object = tpu_cluster.get(r)
                        
                        print(f"Step {log_object['step']}: "
                              f"Loss: {log_object['loss']:.3f}, "
                              f"It/s: {log_object['iterations_per_second']:.1f}, "
                              f"Data Loader Time Frac: {log_object['data_loader_time_fraction']:.3f}")
                        
                        if i % config["autoregressive_diffusion_model"]["train"]["ckpt_freq"] == 0:
                            # Save checkpoint logic here
                            pass
                        
                    except tpu_infra.DeadTPUException as e:
                        print("TPU failure detected, restarting...")
                        del train_func
                        del r
                        tpu_cluster.restart()
                        
                        # Recreate the harnesses after restart
                        factory_on_tpu = tpu_cluster.put([ardm_harness_factory] * n)
                        ardm_harnesses = factory_on_tpu(args, config)
                        train_func = ardm_harnesses.train
                
                t2 = time.time()
                print(f"Training completed. Total time: {t2-t1}, Time per step: {(t2-t1)/steps}")
                
            finally:
                tpu_infra.shutdown()
        elif backend == "gpu":
            # TODO: Implement distributed GPU training
            raise NotImplementedError("Distributed GPU training not implemented for ARDM")
        else:
            # TODO: Implement distributed CPU training
            raise NotImplementedError("Distributed CPU training not implemented for ARDM")
    elif args.mode == "swarm":
        # TODO: Implement swarm training
        raise NotImplementedError("Swarm training not implemented for ARDM")
    else:
        print(f"Mode {args.mode} and backend {backend} is not supported for train_adm")

def reconstruct_image(config, args):
    print(f"Reconstructing image {args.input_image} with config {config} in {args.mode} mode")

    backend = config.get("backend", "cpu")
    if args.mode == "local":
        # TODO: Implement local image reconstruction
        pass
    elif args.mode == "distributed":
        # TODO: Implement distributed image reconstruction
        pass
    elif args.mode == "swarm":
        # TODO: Implement swarm image reconstruction
        pass
    else:
        print(f"Mode {args.mode} is not supported for reconstruct_image")

def sample_video(config, args):
    print(f"Sampling video with config {config} in {args.mode} mode")
    if args.text_prompt:
        print(f"Using text prompt: {args.text_prompt}")
    elif args.image_prompt:
        print(f"Using image prompt: {args.image_prompt}")
    elif args.video_prompt:
        print(f"Using video prompt: {args.video_prompt}")

    def ardm_harness_factory():
        return dar.DiffARHarness(args, config)

    backend = config.get("backend", "cpu")
    if args.mode == "local":
        if backend == "tpu":
            ardm_harness = ardm_harness_factory()
            generated_video = ardm_harness.sample()
            print(f"Generated video shape: {generated_video.shape}")
            # TODO: Implement saving or further processing of the generated video
        elif backend == "gpu":
            # TODO: Implement local GPU sampling
            pass
        else:
            # TODO: Implement local CPU sampling
            pass
    elif args.mode == "distributed":
        # TODO: Implement distributed video sampling
        pass
    elif args.mode == "swarm":
        # TODO: Implement swarm video sampling
        pass
    else:
        print(f"Mode {args.mode} is not supported for sample_video")

if __name__ == "__main__":
    main()
