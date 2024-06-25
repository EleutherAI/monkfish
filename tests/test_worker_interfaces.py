import pytest
import numpy as np
import PIL.Image as Image
import io
import fs.memoryfs
import catfish.lvd.shrd_data_loader as sdl

@pytest.fixture
def mock_fs():
    memory_fs = fs.memoryfs.MemoryFS()
    
    # Create some test images
    for i in range(5):
        img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
        with memory_fs.open(f'image_{i}.jpg', 'wb') as f:
            img.save(f, format='JPEG')
    
    return memory_fs

@pytest.fixture
def image_worker(mock_fs):
    return sdl.ImageWorkerInterface(mock_fs)

def test_init(mock_fs):
    worker = sdl.ImageWorkerInterface(mock_fs)
    assert len(worker.files) == 5
    assert worker.files == ['image_0.jpg', 'image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg']

def test_get_example(image_worker):
    image_array, example_id = image_worker.get_example(0)
    assert isinstance(image_array, np.ndarray)
    assert image_array.shape == (100, 100, 3)
    assert example_id == 0

    # Test looping behavior
    image_array, example_id = image_worker.get_example(5)
    assert example_id == 5
    assert np.array_equal(image_array, image_worker.get_example(0)[0])

def test_list_dir(image_worker):
    files = image_worker.list_dir()
    assert len(files) == 5
    assert files == ['image_0.jpg', 'image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg']

def test_empty_directory():
    empty_fs = fs.memoryfs.MemoryFS()
    with pytest.raises(ValueError, match="The directory is empty. No images to process."):
        sdl.ImageWorkerInterface(empty_fs)

def test_get_example_invalid_id(image_worker):
    # This should not raise an error due to the looping behavior
    image_array, example_id = image_worker.get_example(1000)
    assert example_id == 1000
    assert np.array_equal(image_array, image_worker.get_example(0)[0])