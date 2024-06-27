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
def mock_worker_interface_factory():
    upload_count = multiprocessing.Value('i', 0)
    def factory():
        interface = MagicMock()
        def upload_example_with_count(counter, example):
            with upload_count.get_lock():
                upload_count.value += 1
        interface.upload_example = MagicMock(side_effect=upload_example_with_count)
        return interface
    return factory, upload_count

@pytest.fixture
def mock_shard_interface_factory():
    def factory():
        interface = MagicMock()
        interface.accelerator_to_host = lambda x: [f"data_{i}" for i in range(0,len(x),4)]
        return interface
    return factory

@pytest.fixture
def sharded_uploader(dist_manager_factory, mock_worker_interface_factory, mock_shard_interface_factory):
    dist_manager = dist_manager_factory(1)
    worker_interface_factory, upload_count = mock_worker_interface_factory
    uploader = sdu.ShardedDataUploader(
        worker_interface_factory=worker_interface_factory,
        shard_interface_factory=mock_shard_interface_factory,
        dist_manager=dist_manager,
        workers_per_node=2,
        batch_size=64,
        queue_depth=10
    )
    return uploader, upload_count

@pytest.fixture
def sharded_uploaders(dist_manager_factory, mock_worker_interface_factory, mock_shard_interface_factory):
    worker_interface_factory, upload_count = mock_worker_interface_factory
    uploaders = []
    for i in range(4):
        dist_manager = dist_manager_factory(i)
        uploader = sdu.ShardedDataUploader(
            worker_interface_factory=worker_interface_factory,
            shard_interface_factory=mock_shard_interface_factory,
            dist_manager=dist_manager,
            workers_per_node=2,
            batch_size=64,
            queue_depth=10
        )
        uploaders.append(uploader)
    return uploaders

def test_round_robin_scheduling(sharded_uploader):
    sharded_uploader, _ = sharded_uploader
    
    sharded_uploader.start(0)

    num_steps = 10
    mock_accelerator_data = [1] * sharded_uploader.batch_size  # Dummy data
    last_index = None
    for _ in range(num_steps):
        sharded_uploader.step(mock_accelerator_data)
        sharded_uploader.ack()
        if last_index is not None:
            expected_index = (last_index + sharded_uploader.batch_size // sharded_uploader.nodes) % sharded_uploader.workers_per_node
            assert sharded_uploader.round_robin_index == expected_index, "Round robin index not handled correctly"
        last_index = sharded_uploader.round_robin_index

    sharded_uploader.stop()

def test_worker_shutdown(sharded_uploader):
    sharded_uploader, _ = sharded_uploader

    sharded_uploader.start(0)
    sharded_uploader.stop()

    # Check all workers are no longer active
    for worker in sharded_uploader.workers:
        assert not worker.is_alive(), "Worker did not shut down correctly"

def test_batch_elements_distribution(sharded_uploaders):
    #sharded_uploader, _ = sharded_uploader
    
    for uploader in sharded_uploaders:
        uploader.start(0)

    mock_accelerator_data = [1] * sharded_uploaders[0].batch_size  # Dummy data

    for i, uploader in enumerate(sharded_uploaders):
        processed_count = uploader.step(mock_accelerator_data)
        expected_count = uploader.batch_size // uploader.dist_manager.nodes
        assert processed_count == expected_count, f"Uploader {i} processed incorrect number of elements: expected {expected_count}, got {processed_count}"

    for uploader in sharded_uploaders:
        uploader.stop()

def test_upload_example_count(sharded_uploader):
    uploader, upload_count = sharded_uploader
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

