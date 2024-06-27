import argparse

import catfish.lvd.diffusion_ae as dae
import catfish.lvd.diffusion_ar as dar


import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Sharded/distributed training tool for video generative model")

    # Required arguments
    parser.add_argument("config", help="Path to configuration file (JSON)")
    parser.add_argument("mode", choices=["local", "distributed"], help="Mode of operation: local (single node) or distributed (multinode)")

    # Subparsers for different operations
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Operation to perform")

    # Training diffusion autoencoder
    subparsers.add_parser("train_dae", help="Train the diffusion autoencoder")
    
    # Lifting videos
    lift_parser = subparsers.add_parser("lift", help="Lift videos into the diffusion latent space")
    lift_parser.add_argument("input_videos", nargs="+", help="Input video files")

    # Training autoregressive diffusion model
    subparsers.add_parser("train_adm", help="Train the autoregressive diffusion model")

    # Reconstructing static test image
    reconstruct_parser = subparsers.add_parser("reconstruct", help="Reconstruct a static test image")
    reconstruct_parser.add_argument("input_image", help="Input image file")

    # Sampling
    sample_parser = subparsers.add_parser("sample", help="Sample a video using both models")
    sample_parser.add_argument("--text_prompt", help="Text prompt for sampling")
    sample_parser.add_argument("--image_prompt", help="Image prompt for sampling")
    sample_parser.add_argument("--video_prompt", help="Video prompt for sampling")

    return parser.parse_args()

def read_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()
    config = read_config(args.config)

    # Process the arguments and call appropriate functions
    if args.operation == "train_dae":
        train_diffusion_autoencoder(config, args.mode)
    elif args.operation == "lift":
        lift_videos(config, args.mode, args.input_videos)
    elif args.operation == "train_adm":
        train_autoregressive_diffusion_model(config, args.mode)
    elif args.operation == "reconstruct":
        reconstruct_image(config, args.mode, args.input_image)
    elif args.operation == "sample":
        sample_video(config, args.mode, args.text_prompt, args.image_prompt, args.video_prompt)

def train_diffusion_autoencoder(config, mode):
    print(f"Training diffusion autoencoder with config {config} in {mode} mode")

def lift_videos(config, mode, input_videos):
    print(f"Lifting videos {input_videos} with config {config} in {mode} mode")

def train_autoregressive_diffusion_model(config, mode):
    print(f"Training autoregressive diffusion model with config {config} in {mode} mode")

def reconstruct_image(config, mode, input_image):
    print(f"Reconstructing image {input_image} with config {config} in {mode} mode")

def sample_video(config, mode, text_prompt, image_prompt, video_prompt):
    print(f"Sampling video with config {config} in {mode} mode")
    if text_prompt:
        print(f"Using text prompt: {text_prompt}")
    elif image_prompt:
        print(f"Using image prompt: {image_prompt}")
    elif video_prompt:
        print(f"Using video prompt: {video_prompt}")

if __name__ == "__main__":
    main()