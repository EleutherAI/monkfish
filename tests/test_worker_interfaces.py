import io
import os
import tempfile
import pickle

import pytest
import numpy as np
import PIL.Image as Image
import fs.memoryfs
import catfish.lvd.shrd_data_loader as sdl
import moviepy.editor
import mutagen.mp4
import jax
import jax.numpy as jnp


@pytest.fixture
def mock_image_fs():
    memory_fs = fs.memoryfs.MemoryFS()
    
    # Create some test images
    for i in range(5):
        img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
        with memory_fs.open(f'image_{i}.jpg', 'wb') as f:
            img.save(f, format='JPEG')
    
    return memory_fs

@pytest.fixture
def mock_latent_fs():
    memory_fs = fs.memoryfs.MemoryFS()
    
    # Create some test latent data
    for i in range(5):
        data = (f"String {i}", np.random.rand(10, 10).astype(np.float32))
        with memory_fs.open(f'{i+1}.pkl', 'wb') as f:
            pickle.dump(data, f)
    
    return memory_fs

@pytest.fixture
def mock_video_fs():
    memory_fs = fs.memoryfs.MemoryFS()

    # Create some test videos
    for i in range(3):
        # Create a simple video clip
        duration = 2  # 2 seconds
        fps = 24
        frames = [np.full((100, 100, 3), i*50, dtype=np.uint8) for _ in range(int(duration * fps))]
        
        # Create clip from frames
        clip = moviepy.editor.ImageSequenceClip(frames, fps=fps)
        
        # Use a temporary file to save the video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            clip.write_videofile(temp_path, codec='libx264', audio=False, verbose=False, logger=None)
        
        # Read the video data from the temporary file
        with open(temp_path, 'rb') as f:
            video_data = f.read()
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Write the video data to the memory filesystem
        file_name = f'video_{i}.mp4'
        with memory_fs.open(file_name, 'wb') as f:
            f.write(video_data)
        
        # Add metadata
        with memory_fs.open(file_name, 'rb+') as f:
            mp4 = mutagen.mp4.MP4(f)
            mp4['\xa9des'] = f'Description for video {i}'
            mp4.save(f)

    return memory_fs

@pytest.fixture
def latent_worker(mock_latent_fs):
    return sdl.LatentWorkerInterface(mock_latent_fs)

@pytest.fixture
def mock_dist_manager():
    class MockDistManager:
        def __init__(self):
            self.mesh = jax.sharding.Mesh(np.array([jax.local_devices()]), ('dp','mp'))
        
        def scatter(self, sharding, dtype):
            return lambda x: jax.device_put(x, sharding)
        
        def gather(self):
            return lambda x: np.array(x)
    
    return MockDistManager()

@pytest.fixture
def latent_shard_interface(mock_dist_manager):
    return sdl.LatentShardInterface(mock_dist_manager)

@pytest.fixture
def image_worker(mock_image_fs):
    return sdl.ImageWorkerInterface(mock_image_fs)

@pytest.fixture
def video_worker(mock_video_fs):
    return sdl.VideoWorkerInterface(mock_video_fs)

def test_image_init(mock_image_fs):
    worker = sdl.ImageWorkerInterface(mock_image_fs)
    assert len(worker.files) == 5
    assert worker.files == ['image_0.jpg', 'image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg']

def test_image_get_example(image_worker):
    image_array, example_id = image_worker.get_example(0)
    assert isinstance(image_array, np.ndarray)
    assert image_array.shape == (100, 100, 3)
    assert example_id == 0

    # Test looping behavior
    image_array, example_id = image_worker.get_example(5)
    assert example_id == 5
    assert np.array_equal(image_array, image_worker.get_example(0)[0])

def test_image_list_dir(image_worker):
    files = image_worker.list_dir()
    assert len(files) == 5
    assert files == ['image_0.jpg', 'image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg']

def test_image_empty_directory():
    empty_fs = fs.memoryfs.MemoryFS()
    with pytest.raises(ValueError, match="The directory is empty. No images to process."):
        sdl.ImageWorkerInterface(empty_fs)

def test_image_get_example_invalid_id(image_worker):
    # This should not raise an error due to the looping behavior
    image_array, example_id = image_worker.get_example(1000)
    assert example_id == 1000
    assert np.array_equal(image_array, image_worker.get_example(0)[0])

def test_video_init(mock_video_fs):
    worker = sdl.VideoWorkerInterface(mock_video_fs)
    assert len(worker.files) == 3
    assert worker.files == ['video_0.mp4', 'video_1.mp4', 'video_2.mp4']

def test_video_get_example(video_worker):
    (video_array, description), example_id = video_worker.get_example(0)
    assert isinstance(video_array, np.ndarray)
    assert video_array.shape[1:] == (100, 100, 3)  # (frames, height, width, channels)
    assert description == 'Description for video 0'
    assert example_id == 0

    # Test looping behavior
    (video_array, description), example_id = video_worker.get_example(3)
    assert example_id == 3
    assert np.array_equal(video_array, video_worker.get_example(0)[0][0])

def test_video_list_dir(video_worker):
    files = video_worker.list_dir()
    assert len(files) == 3
    assert files == ['video_0.mp4', 'video_1.mp4', 'video_2.mp4']

def test_video_empty_directory():
    empty_fs = fs.memoryfs.MemoryFS()
    with pytest.raises(ValueError, match="The directory is empty or contains no MP4 files."):
        sdl.VideoWorkerInterface(empty_fs)

def test_latent_init(mock_latent_fs):
    worker = sdl.LatentWorkerInterface(mock_latent_fs)
    assert len(worker.files) == 5
    assert worker.files == ['1.pkl', '2.pkl', '3.pkl', '4.pkl', '5.pkl']

def test_latent_get_example(latent_worker):
    data, example_id = latent_worker.get_example(0)
    assert isinstance(data, tuple)
    assert isinstance(data[0], str)
    assert isinstance(data[1], np.ndarray)
    assert data[1].shape == (10, 10)
    assert example_id == 0

    # Test looping behavior
    data, example_id = latent_worker.get_example(5)
    assert example_id == 5
    
    first_data = latent_worker.get_example(0)[0]
    assert data[0] == first_data[0]  # Compare strings
    np.testing.assert_array_equal(data[1], first_data[1])  # Compare numpy arrays

def test_latent_list_dir(latent_worker):
    files = latent_worker.list_dir()
    assert len(files) == 5
    assert files == ['1.pkl', '2.pkl', '3.pkl', '4.pkl', '5.pkl']

def test_latent_empty_directory():
    empty_fs = fs.memoryfs.MemoryFS()
    with pytest.raises(ValueError, match="The directory is empty or contains no pickle files."):
        sdl.LatentWorkerInterface(empty_fs)

def test_latent_upload_example(latent_worker, mock_latent_fs):
    data = ("Test string", np.random.rand(10, 10).astype(np.float32))
    latent_worker.upload_example(6, data)
    
    assert '6.pkl' in mock_latent_fs.listdir('/')
    
    with mock_latent_fs.open('6.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    assert loaded_data[0] == data[0]
    np.testing.assert_array_equal(loaded_data[1], data[1])

def test_latent_shard_interface_host_to_accelerator(latent_shard_interface):
    local_data = [
        (("String 1", np.random.rand(10, 10).astype(np.float32)), 0),
        (("String 2", np.random.rand(10, 10).astype(np.float32)), 1),
    ]
    
    strings, sharded_array = latent_shard_interface.host_to_accelerator(local_data, 2)
    
    assert strings == ["String 1", "String 2"]
    assert isinstance(sharded_array, jax.Array)
    assert sharded_array.shape == (2, 10, 10)

def test_latent_shard_interface_accelerator_to_host(latent_shard_interface):
    strings = ["String 1", "String 2"]
    array = np.random.rand(2, 10, 10).astype(np.float32)
    sharded_array = jax.device_put(array)
    
    global_data = (strings, sharded_array)
    
    local_data = latent_shard_interface.accelerator_to_host(global_data)
    
    assert len(local_data) == 2
    assert local_data[0][0] == "String 1"
    assert local_data[1][0] == "String 2"
    np.testing.assert_array_equal(local_data[0][1], array[0])
    np.testing.assert_array_equal(local_data[1][1], array[1])

def test_latent_end_to_end(latent_worker, latent_shard_interface):
    # Get examples from worker
    examples = [latent_worker.get_example(i) for i in range(2)]
    
    # Pass through shard interface
    strings, sharded_array = latent_shard_interface.host_to_accelerator(examples, 2)
    
    # Simulate some processing
    processed_array = sharded_array + 1 - 1
    
    # Pass back through shard interface
    processed_data = latent_shard_interface.accelerator_to_host((strings, processed_array))
    
    # Upload processed data
    for i, (data, _) in enumerate(processed_data):
        latent_worker.upload_example(i + 10, data)  # Use new IDs to avoid overwriting
    
    # Verify uploaded data
    for i in range(2):
        uploaded_data, _ = latent_worker.get_example(i + 10)
        assert uploaded_data[0] == strings[i]
        np.testing.assert_array_equal(uploaded_data[1], processed_array[i])