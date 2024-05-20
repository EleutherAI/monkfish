import os
import sys
import numpy as np
import catfish.lvd.gcp_data_loader as gdl

def cleanup_bucket(bucket):
    """Removes all contents from the specified bucket."""
    print("Cleaning up the bucket...")
    blobs = list(bucket.list_blobs())
    for blob in blobs:
        blob.delete()
    print("Bucket cleaned.")

def upload_test_data(uploader):
    """Uploads videos and descriptions."""
    print("Uploading videos and descriptions...")
    uploader.upload_videos()

def test_random_frame(data_loader):
    """Tests random frame extraction."""
    print("Testing random frame extraction...")
    data_loader.start_workers(mode='random_frame', worker_count=2)  # Use 2 threads
    try:
        for _ in range(5):  # Fetch 5 frames
            frame_data = data_loader.get_data()
            if frame_data is not None:
                print("Sampled Frame Data:", frame_data.shape)
            else:
                print("No data returned.")
    finally:
        data_loader.__exit__(None, None, None)

def test_contiguous_video(data_loader):
    """Tests contiguous video extraction."""
    print("Testing contiguous video extraction...")
    data_loader.start_workers(mode='contiguous_video', worker_count=2)  # Use 2 threads
    try:
        for _ in range(2):  # Fetch 2 video datasets
            video_data = data_loader.get_data()
            if video_data is not None:
                print(type(video_data))
                frames, description = video_data
                print(f"Video Description: {description}, Frames Shape: {frames.shape}")
            else:
                print("No video data returned.")
    finally:
        data_loader.__exit__(None, None, None)

def main():
    # Initialize paths and credentials
    credentials_path = 'service-account-key.json'
    bucket_name = 'lvd_data'  # Replace with your actual bucket name
    upload_folder_path = 'dataset'
    metadata_path = 'dataset/descriptions.json'

    """
    # Initialize VideoUploader
    uploader = gdl.VideoUploader(credentials_path, bucket_name, upload_folder_path, metadata_path)

    # Cleanup the bucket first
    cleanup_bucket(uploader.bucket)

    # Upload new test data
    upload_test_data(uploader)
    """

    # Initialize VideoDataLoader
    target_resolution = (720, 1280)  # Example target resolution
    data_loader = gdl.VideoDataLoader(credentials_path, bucket_name, upload_folder_path, target_resolution, metadata_path)
    
    # Test random frame mode
    test_random_frame(data_loader)


    """
    # Test contiguous video mode
    test_contiguous_video(data_loader)
    """

    print("All tests completed successfully.")


if __name__ == "__main__":
    main()
