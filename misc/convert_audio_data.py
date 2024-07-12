import os
import random
import pickle
import numpy as np
import argparse
from pydub import AudioSegment
from scipy.signal import stft

def random_sample_mp3(file_path, duration_ms):
    audio = AudioSegment.from_mp3(file_path)
    if len(audio) <= duration_ms:
        return audio
    start = random.randint(0, len(audio) - duration_ms)
    return audio[start:start + duration_ms]

def create_dataset(input_dir, output_dir, sample_duration, num_buckets, num_samples):
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Randomly select an MP3 file
        mp3_file = random.choice(mp3_files)
        file_path = os.path.join(input_dir, mp3_file)
        
        # Extract a random sample
        audio_sample = random_sample_mp3(file_path, sample_duration)
        
        # Convert to mono and get the raw audio data
        audio_array = np.array(audio_sample.set_channels(1).get_array_of_samples())
        
        # Perform STFT
        _, _, Zxx = stft(audio_array, nperseg=num_buckets)
        
        # Create the tuple
        file_name = os.path.splitext(mp3_file)[0]
        data_tuple = (file_name, np.abs(Zxx))
        
        # Save as pickle file
        output_file = os.path.join(output_dir, f"{i}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(data_tuple, f)
        
        print(f"Processed sample {i+1}/{num_samples}")

def main():
    parser = argparse.ArgumentParser(description="Create a dataset from MP3 files using STFT")
    parser.add_argument("input_dir", help="Path to the input directory containing MP3 files")
    parser.add_argument("output_dir", help="Path to the output directory for pickle files")
    parser.add_argument("--duration", type=int, default=5000, help="Duration of each sample in milliseconds (default: 5000)")
    parser.add_argument("--buckets", type=int, default=256, help="Number of buckets for STFT (default: 256)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate (default: 1000)")

    args = parser.parse_args()

    create_dataset(args.input_dir, args.output_dir, args.duration, args.buckets, args.samples)

if __name__ == "__main__":
    main()
