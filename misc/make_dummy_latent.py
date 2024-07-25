import os
import pickle
import numpy as np
import argparse

def generate_dummy_latent_data(num_samples, latent_dim1, latent_dim2, id_length, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_samples):
        # Generate a 1D integer numpy array as ID
        dummy_id = np.random.randint(0, 256, size=id_length, dtype=np.int32)

        # Generate a 2D dummy numpy array
        dummy_array = np.random.randn(latent_dim1, latent_dim2).astype(np.float32)*0.001

        # Create the tuple
        dummy_data = (dummy_id, dummy_array)

        # Save the tuple to a pickle file
        filename = os.path.join(output_dir, f"{i}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(dummy_data, f)

    print(f"Generated {num_samples} dummy latent samples in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate dummy latent data for LatentWorkerInterface")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of dummy samples to generate")
    parser.add_argument("--latent_dim1", type=int, default=128, help="First dimension of the latent array")
    parser.add_argument("--latent_dim2", type=int, default=8, help="Second dimension of the latent array")
    parser.add_argument("--id_length", type=int, default=8, help="Length of the 1D integer ID array")
    parser.add_argument("--output_dir", type=str, default="../dummy_latent_data", help="Directory to save the dummy data")

    args = parser.parse_args()

    generate_dummy_latent_data(args.num_samples, args.latent_dim1, args.latent_dim2, args.id_length, args.output_dir)

if __name__ == "__main__":
    main()
