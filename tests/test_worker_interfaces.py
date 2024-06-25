
import io
import os
import tempfile

import pytest
import numpy as np
import PIL.Image as Image
import fs.memoryfs
import catfish.lvd.shrd_data_loader as sdl
import moviepy.editor
import mutagen.mp4

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

