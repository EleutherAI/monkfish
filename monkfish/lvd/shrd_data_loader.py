import os
import io
import sys
import time
import pickle

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

import jax
import jax.numpy as jnp
import jax.sharding as shrd

import monkfish.lvd.fs_utils as fs_utils

def sdu_worker(start_index, queue, stop_event, workers_per_node, nodes, 
               worker_interface_cls, fs_init_args):
    counter = start_index

    fs = fs_utils.fs_initializer(fs_init_args)
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
    print(f"sdd worker started with  start_index: {start_index}")
    counter = start_index

    fs = fs_utils.fs_initializer(fs_init_args)
    worker_interface = worker_interface_cls(fs)
    
    while not stop_event.is_set():
        example = worker_interface.get_example(counter)
        try:
            # Try to put the item in the queue with a timeout
            queue.put((example, counter), timeout=0.1)
            counter += workers_per_node * nodes
        except multiprocessing.queues.Full:
            # If the queue is full, just continue to the next iteration
            continue
    queue.close()
    print("worker finished")


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

        while any(not queue.empty() for queue in self.queues):
            for queue in self.queues:
                try:
                    queue.get_nowait()
                except multiprocessing.queues.Empty:
                    continue
            # Wait a bit to make sure nothing else is added to the queues
            # SDD will deadlock otherwise
            #TODO: Make this less hacky
            time.sleep(0.3)
        
        
        for worker in self.workers:
            worker.join()
        
        self.queues = None
        self.stop_event = None

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
        normed_image_array = image_array/255 - 0.5
        return normed_image_array, example_id

    def list_dir(self):
        """
        List all examples in the folder.
        """
        return self.files

    def upload_example(self, example_id, image_data):
        """
        Upload a processed image back to the filesystem.
        :param example_id: Integer, unique identifier for the example.
        :param image_data: Numpy array, the image data to be uploaded.
        """
        file_name = f'image_{example_id}.png'  # Define the file name pattern
        
        # Convert the numpy array back to an image
        image = Image.fromarray((image_data * 255 + 0.5).astype('uint8'))  # Undo normalization
        
        # Save the image to a bytes buffer
        with io.BytesIO() as buffer:
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Upload to filesystem
            with self.fs.open(file_name, 'wb') as fs_file:
                fs_file.write(buffer.getvalue())
                print(f'Uploaded {file_name} successfully.')

class ImageShardInterface:
    """Interface to image data, folder is a dataset, 1 train example per file"""
    def __init__(self, dist_manager):
        self.dist_manager = dist_manager
        pass

    def host_to_accelerator(self, local_data, batch_size):
        #TODO: Remove dummy data and generalize properly to multinode
        shape = (batch_size, 3, 512, 256)
        arrays = [x[0] for x in local_data]
        print("Shape:",shape, np.stack(arrays).shape)
        #print("Shapes:", shape, local_data.shape)
        np_array = np.transpose(np.stack(arrays).astype(np.float32),(0, 3, 2, 1))
        mesh = self.dist_manager.mesh
        p_spec = shrd.PartitionSpec("dp")
        sharding = shrd.NamedSharding(mesh, p_spec)
        jax_array = jnp.array(np_array)
        scatter_fn = self.dist_manager.scatter(sharding, jnp.float32)
        sharded_array = scatter_fn(jax_array)
        return sharded_array
    
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
    """Interface to Latent data, folder is a dataset, 1 train example per file"""

    def __init__(self, fs):
        self.fs = fs
        self.files = sorted([f for f in self.fs.listdir('/') if f.lower().endswith('.pkl')])
        if not self.files:
            raise ValueError("The directory is empty or contains no pickle files.")

    def get_example(self, example_id):
        num_files = len(self.files)
        file_index = example_id % num_files
        file_name = self.files[file_index]
        
        with self.fs.open(file_name, 'rb') as latent_file:
            data = pickle.load(latent_file)
        
        return data, example_id

    def list_dir(self):
        """
        List all examples in the folder.
        """
        return self.files

    def upload_example(self, example_id, data):
        """
        Upload a processed latent example back to the filesystem.
        """
        file_name = f'{example_id}.pkl'
        
        with self.fs.open(file_name, 'wb') as fs_file:
            pickle.dump(data, fs_file)

class LatentShardInterface:
    """Interface to Latent data for sharding"""

    def __init__(self, dist_manager):
        self.dist_manager = dist_manager

    def host_to_accelerator(self, local_data, batch_size):
        tokens = [item[0][0] for item in local_data]  # Extracting tokens vectors
        arrays = [item[0][1] for item in local_data]  # Extracting numpy arrays
        
        # Stack the arrays
        jax_token_data = jnp.stack(tokens)
        jax_array_data = jnp.stack(arrays)
        
        mesh = self.dist_manager.mesh
        p_spec = shrd.PartitionSpec("dp")
        sharding = shrd.NamedSharding(mesh, p_spec)
        scatter_fn = self.dist_manager.scatter(sharding, jnp.float32)
        sharded_token_data = scatter_fn(jax_token_data)
        sharded_array_data = scatter_fn(jax_array_data)
        
        return sharded_token_data, sharded_array_data
    
    def accelerator_to_host(self, global_data):
        sharded_token_data, sharded_array_data = global_data
        
        # Gather the sharded arrays back to host
        gather_fn = self.dist_manager.gather()
        np_token_data = np.array(gather_fn(sharded_token_data))
        np_array_data = np.array(gather_fn(sharded_array_data))
        
        # Combine tokens and arrays back into the original format
        local_data = [([tokens], array) for tokens, array in zip(np_token_data, np_array_data)]
        
        return local_data

