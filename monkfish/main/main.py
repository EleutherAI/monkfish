import os
import fs_util

import argparse
import json
import multiprocessing


import monkfish.lvd.diffusion_ae as dae
import monkfish.lvd.diffusion_ar as dar
import monkfish.lvd.fs_utils as fs_utils

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
    clear_dir_parser = subparsers.add_parser("clear_fs", help="Deletes a specified directory")
    clear_dir_parser.add_argument("--fs_type", type=str, help="file system type")
    clear_dir_parser.add_argument("--root_dir", type=str, help="root_dir")
    
    # List dir 
    clear_fs_parser = subparsers.add_parser("clear_fs", help="List a specified directory")
    clear_fs_parser.add_argument("--fs_type", type=str, help="file system type")
    clear_fs_parser.add_argument("--root_dir", type=str, help="root_dir")

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
        train_autoregressive_diffusion_model(config, args)
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

    try:
        if filesystem.exists('/'):
            filesystem.removetree('/')
            print(f"Cleared {fs_type} directory: {root_dir}")
        else:
            print(f"Directory does not exist: {root_dir}")
    except Exception as e:
        print(f"Error clearing {fs_type} directory: {e}")

def list_filesystem(config, args):
    filesystem, fs_type, root_dir = setup_filesystem(config, args)
    if not filesystem:
        return

    def print_dir(path, level=0):
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(path) or path}/")
        for item in filesystem.scandir(path):
            if item.is_dir:
                print_dir(item.path, level + 1)
            else:
                print(f"{indent}    {item.name}")

    try:
        if filesystem.exists('/'):
            print_dir('/')
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
            dae_harness.train()
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
            # TODO: Implement distributed TPU training
            pass
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

    def ardm_harness_factory():
        return dar.DiffARHarness(
            args,
            config
        )

    backend = config.get("backend", "cpu")
    if args.mode == "local":
        if backend == "tpu":
            ardm_harness = ardm_harness_factory()
            ardm_harness.train()
        elif backend == "gpu":
            # TODO: Implement local GPU training
            pass
        else:
            # TODO: Implement local CPU training
            pass
    elif args.mode == "distributed":
        if backend == "tpu":
            # TODO: Implement distributed TPU training
            pass
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

    backend = config.get("backend", "cpu")
    if args.mode == "local":
        # TODO: Implement local video sampling
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
