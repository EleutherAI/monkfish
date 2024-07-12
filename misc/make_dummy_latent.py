import os
import pickle
import numpy as np
import argparse

def generate_dummy_latent_data(num_samples, latent_dim1, latent_dim2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_samples):
        # Generate a dummy string
        dummy_string = f"This is dummy text for sample {i}"

        # Generate a 2D dummy numpy array
        dummy_array = np.random.randn(latent_dim1, latent_dim2).astype(np.float32)

        # Create the tuple
        dummy_data = (dummy_string, dummy_array)

        # Save the tuple to a pickle file
        filename = os.path.join(output_dir, f"{i}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(dummy_data, f)

    print(f"Generated {num_samples} dummy latent samples in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate dummy latent data for LatentWorkerInterface")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of dummy samples to generate")
    parser.add_argument("--latent_dim1", type=int, default=32, help="First dimension of the latent array")
    parser.add_argument("--latent_dim2", type=int, default=64, help="Second dimension of the latent array")
    parser.add_argument("--output_dir", type=str, default="dummy_latent_data", help="Directory to save the dummy data")

    args = parser.parse_args()

    generate_dummy_latent_data(args.num_samples, args.latent_dim1, args.latent_dim2, args.output_dir)

if __name__ == "__main__":
    main()
