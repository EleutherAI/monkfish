import os
import io
import sys
import time

import numpy as np

import multiprocessing
import multiprocessing.queues

import fs
import fs.osfs
import fs_gcsfs

import google.cloud
import google.oauth2

import PIL.Image as Image

import mutagen.mp4
import cv2
import tempfile


def gcp_filesystem(bucket_name, root_path, credentials_path):
    if not credentials_path:
        raise ValueError("Credentials path must be provided for authentication.")

    # Load credentials from the service account file
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(credentials_path)

    # Initialize the GCS client with the provided credentials
    client = google.cloud.storage.Client(credentials=credentials, project=credentials.project_id)

    # Create a filesystem instance using the GCS client
    google_cloud_storage_fs = fs_gcsfs.GCSFS(bucket_name=bucket_name, root_path=root_path, client=client)
    return google_cloud_storage_fs

def os_filesystem(root_path):
    local_fs = fs.osfs.OSFS(root_path)
    return local_fs

def fs_initializer(args):
    fs_type = args.get('fs_type')

    if fs_type == 'gcp':
        bucket_name = args.get('bucket_name')
        root_path = args.get('root_path', '')
        credentials_path = args.get('credentials_path')

        if not bucket_name:
            raise ValueError("Bucket name must be provided for GCP filesystem.")
        if not credentials_path:
            raise ValueError("Credentials path must be provided for GCP filesystem.")

        return gcp_filesystem(bucket_name, root_path, credentials_path)

    elif fs_type == 'os':
        root_path = args.get('root_path')

        if not root_path:
            raise ValueError("Root path must be provided for OS filesystem.")

        return os_filesystem(root_path)

    else:
        raise ValueError(f"Unsupported filesystem type: {fs_type}")

def sdu_worker(start_index, queue, stop_event, workers_per_node, nodes, 
               worker_interface_cls, fs_init_args):
    counter = start_index

    fs = fs_initializer(fs_init_args)
    worker_interface = worker_interface_cls(fs)
    
    while not stop_event.is_set():
        try:
            # Add a timeout to allow checking stop_event
            example, _ = queue.get(timeout=0.2)
            worker_interface.upload_example(counter, example)
            counter += workers_per_node * nodes
        except multiprocessing.queues.Empty:
            continue
    
    # Drain remaining queue once stop signal is sent
    while not queue.empty():
        example, _ = queue.get()
        worker_interface.upload_example(counter, example)
        counter += workers_per_node * nodes

class ShardedDataUploader:
    def __init__(self, worker_fs_args, worker_interface_factory, shard_interface_factory, dist_manager, 
                 workers_per_node=1, batch_size=32, queue_depth=5):
        assert batch_size % dist_manager.nodes == 0

        # Initialize attributes
        self.workers_per_node = workers_per_node
        self.batch_size = batch_size
        self.queue_depth = queue_depth
        self.dist_manager = dist_manager
        self.pid = dist_manager.pid
        self.nodes = dist_manager.nodes
        self.worker_interface_factory = worker_interface_factory
        self.shard_interface_factory = shard_interface_factory
        self.worker_fs_args = worker_fs_args  # New attribute

        self.shard_interface = self.shard_interface_factory()

        self.counter = None
        self.round_robin_index = None

        self.stop_event = None
        self.workers = []
        self.queues = []

        self.processed = False

    def start(self, counter):
        self.counter = counter
        self.round_robin_index = 0

        self.stop_event = multiprocessing.Event()
        for i in range(self.workers_per_node):
            start_index = self.counter + i*self.nodes + self.pid

            queue = multiprocessing.Queue(maxsize=self.queue_depth)
            worker = multiprocessing.Process(
                target=sdu_worker, args=(
                    start_index, queue, self.stop_event, 
                    self.workers_per_node, self.nodes, self.worker_interface_factory, self.worker_fs_args)
            )

            self.queues.append(queue)
            self.workers.append(worker)
            worker.start()
        
        self.processed = True

    def stop(self):
        self.stop_event.set()

        # Wait for queues to drain before shutting down workers
        while any(not queue.empty() for queue in self.queues):
            time.sleep(0.1)
        
        for worker in self.workers:
            worker.join()
        
        self.queues = None
        self.stop_event = None

    def step(self, accelerator_data):
        assert self.processed

        local_batch_data = self.shard_interface.accelerator_to_host(accelerator_data)

        round_robin_index = self.round_robin_index
        for i, data in enumerate(local_batch_data):
            sub_batch_index = self.round_robin_index + i
            round_robin_index = sub_batch_index % self.workers_per_node

            self.queues[round_robin_index].put((data, None))

        self.processed = False

        return len(local_batch_data)

    def ack(self):
        assert (not self.processed)
        self.counter += self.batch_size
        self.round_robin_index += self.batch_size // self.nodes
        self.round_robin_index %= self.workers_per_node
        self.processed = True

def sdd_worker(start_index, queue, stop_event, workers_per_node, nodes, 
               worker_interface_cls, fs_init_args):
    counter = start_index

    fs = fs_initializer(fs_init_args)
    worker_interface = worker_interface_cls(fs)
    
    while(not stop_event.is_set()):
        example = worker_interface.get_example(counter)
        queue.put((example, counter))
        counter += workers_per_node*nodes

class ShardedDataDownloader:
    def __init__(self, worker_fs_args, worker_interface_cls, shard_interface_factory, dist_manager, 
                 workers_per_node=1, batch_size=32, queue_depth=5):
        assert batch_size % dist_manager.nodes == 0

        #Start workers
        self.workers_per_node = workers_per_node
        self.batch_size = batch_size
        self.queue_depth = queue_depth
        self.dist_manager = dist_manager
        self.pid = dist_manager.pid
        self.nodes = dist_manager.nodes
        self.worker_interface_cls = worker_interface_cls
        self.worker_fs_args = worker_fs_args
        self.shard_interface_factory = shard_interface_factory

        self.shard_interface = self.shard_interface_factory()

        self.counter = None 
        self.round_robin_index = None

        self.stop_event = None
        self.workers = []
        self.queues = []

        self.processed = False
    
    def start(self, counter):
        self.counter = counter
        self.round_robin_index = 0

        self.stop_event = multiprocessing.Event()
        for i in range(self.workers_per_node):
            start_index = self.counter + i*self.nodes + self.pid

            queue = multiprocessing.Queue(maxsize=self.queue_depth)
            worker = multiprocessing.Process(
                target=sdd_worker, args=(
                    start_index, queue, self.stop_event, 
                    self.workers_per_node, self.nodes, self.worker_interface_cls, self.worker_fs_args)
            )

            self.queues.append(queue)
            self.workers.append(worker)
            worker.start()
        
        self.processed = True
    
    def stop(self):
        self.stop_event.set() 

        # Empty queues to unblock workers trying to put items
        #Note,there is still a chance this fails under
        #specific circumstances because of a race condition, 
        #TODO: fix this
        while any(not queue.empty() for queue in self.queues):
            for queue in self.queues:
                try:
                    queue.get_nowait()
                except multiprocessing.queues.Empty:
                    continue
        
        for worker in self.workers:
            worker.join()
        
        self.queues = None
        self.stop_event = None

    def _worker(self, start_index, queue, stop_event):
        counter = start_index
        worker_interface = self.worker_interface_factory()
        
        while(not stop_event.is_set()):
            example = worker_interface.get_example(counter)
            queue.put((example, counter))
            counter += self.workers_per_node*self.nodes

    def step(self):
        assert self.processed

        round_robin_index = self.round_robin_index
        local_batch_data = []
        local_batch_ids = []
        for i in range(self.batch_size // self.nodes):
            sub_batch_index = self.round_robin_index + i
            round_robin_index = sub_batch_index % self.workers_per_node

            data, data_id = self.queues[round_robin_index].get()
            local_batch_data.append(data)
            local_batch_ids.append(data_id)
            
            #Verify expected id
            expected_id = self.counter + sub_batch_index*self.nodes + self.pid
            assert data_id == expected_id
            
        accelerator_data = self.shard_interface.host_to_accelerator(local_batch_data, self.batch_size)
        
        self.processed = False

        return accelerator_data

    def ack(self):
        assert (not self.processed)
        self.counter += self.batch_size
        self.round_robin_index += self.batch_size // self.nodes
        self.round_robin_index %= self.workers_per_node
        self.processed = True

class ImageWorkerInterface:
    """Interface to image data, the folder is a dataset, 1 train example per file."""

    def __init__(self, fs):
        self.fs = fs
        self.files = sorted(self.fs.listdir('/'))  # Assuming the images are in the root directory of the filesystem
        if not self.files:
            raise ValueError("The directory is empty. No images to process.")

    def get_example(self, example_id):
        """
        Fetch an image, process it, and handle looping through images.
        `example_id` is used to select the image file.
        The returned image data will be processed into a numpy array.
        """
        num_files = len(self.files)
        file_index = example_id % num_files  # Ensure looping over the images
        file_name = self.files[file_index]
        
        with self.fs.open(file_name, 'rb') as image_file:
            image_data = image_file.read()
        
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        return image_array, example_id

    def list_dir(self):
        """
        List all examples in the folder.
        """
        return self.files

    def upload_example(self, example_id, image_data):
        """ Optionally implement this if needed to upload processed images back to the filesystem. """
        pass

class ImageShardInterface:
    """Interface to image data, folder is a dataset, 1 train example per file"""
    def __init__(self, dist_manager):
        pass

    def host_to_accelerator(self, local_data, batch_size):
        return "weeb"
    
    def accelerator_to_host(self, global_data):
        pass


class VideoWorkerInterface:
    """Interface to Video data, folder is a dataset, 1 train example per file"""

    def __init__(self, fs):
        self.fs = fs
        self.files = sorted([f for f in self.fs.listdir('/') if f.lower().endswith('.mp4')])
        if not self.files:
            raise ValueError("The directory is empty or contains no MP4 files.")

    def get_example(self, example_id):
        """
        Fetch a video, process it, and handle looping through videos.
        `example_id` is used to select the video file.
        Returns a tuple containing the video as a numpy array and its text description.
        """
        num_files = len(self.files)
        file_index = example_id % num_files  # Ensure looping over the videos
        file_name = self.files[file_index]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name
            
            # Read the file into the temporary file
            with self.fs.open(file_name, 'rb') as video_file:
                temp_file.write(video_file.read())
        
        try:
            # Read metadata
            mp4 = mutagen.mp4.MP4(temp_path)
            description = mp4.get('\xa9des', [''])[0]  # '©des' is the iTunes description tag
            
            # Read video data with OpenCV
            cap = cv2.VideoCapture(temp_path)
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            video_array = np.array(frames)
            
            return (video_array, description), example_id
        
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    def list_dir(self):
        """
        List all examples in the folder.
        """
        return self.files

    def upload_example(self, example_id, video_array, description):
        """
        Upload a processed video back to the filesystem with its description as metadata.
        """
        file_name = f'video_{example_id}.mp4'
        
        # Create an in-memory buffer
        buffer = io.BytesIO()
        
        # Write video to buffer using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter()
        out.open(buffer, fourcc, 30, (video_array.shape[2], video_array.shape[1]), True)
        for frame in video_array:
            out.write(frame)
        out.release()
        
        # Add metadata
        buffer.seek(0)
        mp4 = MP4(buffer)
        mp4['\xa9des'] = description
        mp4.save(buffer)
        
        # Upload to filesystem
        buffer.seek(0)
        with self.fs.open(file_name, 'wb') as fs_file:
            fs_file.write(buffer.getvalue())

class VideoShardInterface:
    def host_to_accelerator(self, local_data, batch_size):
        pass
    
    def accelerator_to_host(self, global_data):
        pass
    

class LatentWorkerInterface:
    pass

class LatentShardInterface:
    def list_dir():
        #List all examples in folder
        pass


