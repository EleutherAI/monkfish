import pytest
from unittest.mock import MagicMock, patch
#from multiprocessing import Event, Queue
import catfish.lvd.shrd_data_loader as sdl

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
    def factory():
        interface = MagicMock()
        interface.get_example = lambda x: (f"data_{x}", x)  # Return both data and ID
        return interface
    return factory

@pytest.fixture
def mock_shard_interface_factory():
    def factory():
        interface = MagicMock()
        interface.host_to_accelerator = lambda x, _: x  # Just return the input
        return interface
    return factory

@pytest.fixture
def sharded_downloader(dist_manager_factory, mock_worker_interface_factory, mock_shard_interface_factory):
    dist_manager = dist_manager_factory(1)
    downloader = sdl.ShardedDataDownloader(
        worker_interface_factory=mock_worker_interface_factory,
        shard_interface_factory=mock_shard_interface_factory,
        dist_manager=dist_manager,
        workers_per_node=2,
        batch_size=64,
        queue_depth=10
    )
    return downloader

@pytest.fixture
def sharded_downloaders(dist_manager_factory, mock_worker_interface_factory, mock_shard_interface_factory):
    downloaders = []
    for i in range(4):
        dist_manager = dist_manager_factory(i)
        downloader = sdl.ShardedDataDownloader(
            worker_interface_factory=mock_worker_interface_factory,
            shard_interface_factory=mock_shard_interface_factory,
            dist_manager=dist_manager,
            workers_per_node=2,
            batch_size=64,
            queue_depth=10
        )
        downloaders.append(downloader)
    return downloaders

"""
def test_local_batch_contiguity(sharded_downloader):
    sharded_downloader.start(0)
    # Assuming step method is updated to store processed batch data
    processed_data = sharded_downloader.step()

    # Check data contiguity by batch
    expected_data = ["processed_data_" + str(i) for i in range(sharded_downloader.batch_size)]
    assert processed_data == expected_data, "Batch data is not contiguous or incorrect"

    sharded_downloader.stop()
"""

def test_round_robin_scheduling(sharded_downloader):
    sharded_downloader.start(0)

    num_steps = 10
    last_index = None
    for _ in range(num_steps):
        sharded_downloader.step()
        sharded_downloader.ack()
        if last_index is not None:
            expected_index = (last_index + sharded_downloader.batch_size // sharded_downloader.nodes) % sharded_downloader.workers_per_node
            assert sharded_downloader.round_robin_index == expected_index, "Round robin index not handled correctly"
        last_index = sharded_downloader.round_robin_index

    sharded_downloader.stop()

def test_worker_shutdown(sharded_downloader):
    sharded_downloader.start(0)
    sharded_downloader.stop()

    # Check all workers are no longer active
    for worker in sharded_downloader.workers:
        assert not worker.is_alive(), "Worker did not shut down correctly"
    
def test_batch_elements_distribution(sharded_downloaders):
    for downloader in sharded_downloaders:
        downloader.start(0)

    all_batches = []
    for downloader in sharded_downloaders:
        batch = downloader.step()
        all_batches.append([item[1] for item in batch])  # Extract only the IDs

    for i, downloader in enumerate(sharded_downloaders):
        batch_size = downloader.batch_size
        expected_elements = set(range(downloader.dist_manager.pid, batch_size, downloader.dist_manager.nodes))
        processed_elements = set(all_batches[i])
        print(f"Downloader {i}:")
        print(f"Expected: {sorted(list(expected_elements))}")
        print(f"Processed: {sorted(list(processed_elements))}")

        assert processed_elements == expected_elements, f"Downloader {i} elements mismatch: expected {expected_elements}, got {processed_elements}"

    for downloader in sharded_downloaders:
        downloader.stop()

# Additional tests can be written to simulate failure scenarios, e.g., worker fails to fetch data