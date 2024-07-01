import time
import pytest
import multiprocessing
from unittest.mock import MagicMock, patch
import catfish.lvd.shrd_data_loader as sdu

@pytest.fixture
def mock_dist_manager():
    manager = MagicMock()
    manager.nodes = 4
    manager.pid = 1
    return manager

@pytest.fixture
def dist_manager_factory():
    def _manager(pid):
        manager = MagicMock()
        manager.nodes = 4  # Assuming there are 4 nodes in total
        manager.pid = pid
        return manager
    return _manager

@pytest.fixture
def mock_worker_interface_cls():
    upload_count = multiprocessing.Value('i', 0)
    class MockWorkerInterface:
        def __init__(self, fs):
            self.fs = fs
        def upload_example(self, counter, example):
            with upload_count.get_lock():
                upload_count.value += 1
    return MockWorkerInterface, upload_count

@pytest.fixture
def mock_shard_interface_factory():
    def factory():
        interface = MagicMock()
        interface.accelerator_to_host = lambda x: [f"data_{i}" for i in range(0,len(x),4)]
        return interface
    return factory

@pytest.fixture
def mock_fs_init_args():
    return {
        'fs_type': 'os',
        'root_path': '/tmp/test_data'
    }

@pytest.fixture
def sharded_uploader(dist_manager_factory, mock_worker_interface_cls, mock_shard_interface_factory, mock_fs_init_args):
    dist_manager = dist_manager_factory(1)
    worker_interface_cls, upload_count = mock_worker_interface_cls
    uploader = sdu.ShardedDataUploader(
        worker_fs_args=mock_fs_init_args,
        worker_interface_factory=worker_interface_cls,
        shard_interface_factory=mock_shard_interface_factory,
        dist_manager=dist_manager,
        workers_per_node=2,
        batch_size=64,
        queue_depth=10
    )
    return uploader, upload_count

@pytest.fixture
def sharded_uploaders(dist_manager_factory, mock_worker_interface_cls, mock_shard_interface_factory, mock_fs_init_args):
    worker_interface_cls, upload_count = mock_worker_interface_cls
    uploaders = []
    for i in range(4):
        dist_manager = dist_manager_factory(i)
        uploader = sdu.ShardedDataUploader(
            worker_fs_args=mock_fs_init_args,
            worker_interface_factory=worker_interface_cls,
            shard_interface_factory=mock_shard_interface_factory,
            dist_manager=dist_manager,
            workers_per_node=2,
            batch_size=64,
            queue_depth=10
        )
        uploaders.append(uploader)
    return uploaders, upload_count

@patch('catfish.lvd.shrd_data_loader.fs_initializer')
def test_round_robin_scheduling(mock_fs_initializer, sharded_uploader):
    uploader, _ = sharded_uploader
    mock_fs = MagicMock()
    mock_fs_initializer.return_value = mock_fs
    
    uploader.start(0)

    num_steps = 10
    mock_accelerator_data = [1] * uploader.batch_size  # Dummy data
    last_index = None
    for _ in range(num_steps):
        uploader.step(mock_accelerator_data)
        uploader.ack()
        if last_index is not None:
            expected_index = (last_index + uploader.batch_size // uploader.nodes) % uploader.workers_per_node
            assert uploader.round_robin_index == expected_index, "Round robin index not handled correctly"
        last_index = uploader.round_robin_index

    uploader.stop()

@patch('catfish.lvd.shrd_data_loader.fs_initializer')
def test_worker_shutdown(mock_fs_initializer, sharded_uploader):
    uploader, _ = sharded_uploader
    mock_fs = MagicMock()
    mock_fs_initializer.return_value = mock_fs

    uploader.start(0)
    uploader.stop()

    # Check all workers are no longer active
    for worker in uploader.workers:
        assert not worker.is_alive(), "Worker did not shut down correctly"

@patch('catfish.lvd.shrd_data_loader.fs_initializer')
def test_batch_elements_distribution(mock_fs_initializer, sharded_uploaders):
    uploaders, _ = sharded_uploaders
    mock_fs = MagicMock()
    mock_fs_initializer.return_value = mock_fs
    
    for uploader in uploaders:
        uploader.start(0)

    mock_accelerator_data = [1] * uploaders[0].batch_size  # Dummy data

    for i, uploader in enumerate(uploaders):
        processed_count = uploader.step(mock_accelerator_data)
        expected_count = uploader.batch_size // uploader.dist_manager.nodes
        assert processed_count == expected_count, f"Uploader {i} processed incorrect number of elements: expected {expected_count}, got {processed_count}"

    for uploader in uploaders:
        uploader.stop()

@patch('catfish.lvd.shrd_data_loader.fs_initializer')
def test_upload_example_count(mock_fs_initializer, sharded_uploader):
    uploader, upload_count = sharded_uploader
    mock_fs = MagicMock()
    mock_fs_initializer.return_value = mock_fs

    uploader.start(0)

    mock_accelerator_data = [1] * uploader.batch_size  # Dummy data
    processed_count = uploader.step(mock_accelerator_data)

    # Wait for all uploads to complete
    max_wait = 5  # Maximum wait time in seconds
    start_time = time.time()
    while upload_count.value < processed_count:
        if time.time() - start_time > max_wait:
            pytest.fail("Timeout waiting for uploads to complete")
        time.sleep(0.1)

    uploader.stop()

    number_of_examples_uploaded = upload_count.value
    expected_calls = uploader.batch_size // uploader.dist_manager.nodes

    assert number_of_examples_uploaded == expected_calls, f"upload_example not called expected number of times. Expected {expected_calls}, got {number_of_examples_uploaded}"
