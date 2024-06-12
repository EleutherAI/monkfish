import json
import cv2
import numpy as np
import os
import tempfile
import pickle
from queue import Queue
from threading import Thread
import google.cloud.storage as gcs

class VideoDataLoader:
    def __init__(self, credentials_path, bucket_name, upload_folder_path, target_resolution=(144, 256), pkl_folder_path="processed_videos"):
        self.client = gcs.Client.from_service_account_json(credentials_path)
        self.bucket = self.client.bucket(bucket_name)
        self.upload_folder_path = upload_folder_path
        self.pkl_folder_path = pkl_folder_path
        self.queue = Queue(maxsize=3)
        self.workers = []
        self.active = True
        self.target_resolution = target_resolution
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        for worker in self.workers:
            worker.join()

    def start_workers(self, mode='random_frame', worker_count=4):
        if mode == 'random_frame':
            self.workers = [Thread(target=self._worker_random_frame) for _ in range(worker_count)]
        elif mode == 'contiguous_video':
            self.workers = [Thread(target=self._worker_contiguous_video) for _ in range(worker_count)]
        elif mode == 'array_tuple':
            self.workers = [Thread(target=self._worker_array_tuple) for _ in range(worker_count)]

        for worker in self.workers:
            worker.start()

    def _download_video(self, temp_local_filename):
        """Download a random .mp4 video file safely into a given file path and return both the file path and the original video file name."""
        try:
            blobs = list(self.bucket.list_blobs())
            mp4_blobs = [blob for blob in blobs if blob.name.endswith('.mp4')]
            if mp4_blobs:
                blob = np.random.choice(mp4_blobs)
                blob.download_to_filename(temp_local_filename)
                return temp_local_filename, blob.name  # Return the path and the original video name
        except Exception as e:
            print(f"Failed to download video: {e}")
            return None, None

    
    def _resize_frame(self, frame):
        """Resize the frame to the target resolution."""
        return cv2.resize(frame, self.target_resolution)

    def _worker_random_frame(self):
        while self.active:
            with tempfile.NamedTemporaryFile() as tmp_file:
                temp_local_filename = tmp_file.name
                video_path, video_name = self._download_video(temp_local_filename)  # This now unpacks both returned values
                if video_path:  # Ensure video_path is not None
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Failed to open video: {video_path}")
                        continue  # Continue to the next iteration

                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count > 0:
                        random_frame_index = np.random.randint(0, frame_count)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
                        ret, frame = cap.read()
                        if ret:
                            resized_frame = self._resize_frame(frame)
                            self.queue.put(np.array(resized_frame))
                    else:
                        print(f"No frames available in video: {video_path}")
                    
                    cap.release()

    def _fetch_video_description(self, video_file):
        """Fetch the description from a JSON file associated with the video."""
        metadata_file = video_file + '.json'
        blob = self.bucket.blob(metadata_file)
        try:
            json_data = blob.download_as_string()
            metadata = json.loads(json_data)
            return metadata.get('description', 'No description available')
        except Exception as e:
            print(f"Failed to download or parse metadata for {video_file}: {e}")
            return 'No description available'

    def _worker_contiguous_video(self):
        while self.active:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                temp_local_filename = tmp_file.name
                video_path, video_name = self._download_video(temp_local_filename)  # Receive both path and name
                if video_path and video_name:
                    description = self._fetch_video_description(video_name)
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret or len(frames) >= 100:
                            break
                        resized_frame = self._resize_frame(frame)
                        frames.append(resized_frame)
                    if frames:
                        self.queue.put((np.array(frames), description))
                    cap.release()

                # Cleanup the temporary file after processing
                os.unlink(video_path)

    def _worker_array_tuple(self):
        """Worker function to download and deserialize .pkl files containing (description, array) tuples."""
        while self.active:
            blobs = list(self.bucket.list_blobs(prefix=self.pkl_folder_path))
            if blobs:
                blob = np.random.choice(blobs)  # Randomly pick a blob to process
                _, temp_local_filename = tempfile.mkstemp(suffix='.pkl')
                blob.download_to_filename(temp_local_filename)
                with open(temp_local_filename, 'rb') as f:
                    description, array = pickle.load(f)
                self.queue.put((description, array))
                os.unlink(temp_local_filename)  # Clean up the temporary file

    def get_data(self):
        return self.queue.get() #if not self.queue.empty() else None


class VideoUploader:
    def __init__(self, credentials_path, bucket_name, upload_folder_path, metadata_path):
        """Initialize the uploader with GCP credentials, bucket details, and metadata path."""
        self.client = gcs.Client.from_service_account_json(credentials_path)
        self.bucket = self.client.bucket(bucket_name)
        self.upload_folder_path = upload_folder_path
        self.metadata_path = metadata_path
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def upload_videos(self):
        """Upload all video files and their descriptions from the specified folder to the GCP bucket."""
        for video_file in os.listdir(self.upload_folder_path):
            if video_file.endswith(('.mp4', '.avi', '.mov')):  # Extendable for other video formats
                video_path = os.path.join(self.upload_folder_path, video_file)
                self._upload_video(video_path)
                self._upload_description(video_file)

    def _upload_description(self, video_file):
        """Upload a description file for each video."""
        description_content = json.dumps({'description': self.metadata.get(video_file, 'No description available')})
        blob = self.bucket.blob(video_file + '.json')
        blob.upload_from_string(description_content)
        print(f"Uploaded description for {video_file}")

    def _upload_video(self, video_path):
        """Helper function to upload a single video file to the bucket."""
        blob_name = os.path.basename(video_path)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(video_path)
        print(f"Uploaded {video_path} to {self.bucket.name}/{blob_name}")

class LatentUploader:
    def __init__(self, credentials_path, bucket_name, target_folder):
        """Initialize the uploader with GCP credentials and bucket details."""
        self.client = gcs.Client.from_service_account_json(credentials_path)
        self.bucket = self.client.bucket(bucket_name)
        self.target_folder = target_folder

    def upload(self, description, array, file_name):
        """Serialize and upload a (description, array) pair as a .pkl file."""
        # Serialize the (description, array) tuple
        data = pickle.dumps((description, array))
        
        # Determine the full path for the blob
        blob_path = os.path.join(self.target_folder, f"{file_name}.pkl")
        
        # Create a new blob and upload the data
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(data)
        print(f"Uploaded {blob_path} to {self.bucket.name}")