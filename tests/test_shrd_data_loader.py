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
def mock_worker_interface():
    interface = MagicMock()
    interface.get_example = MagicMock(side_effect=lambda x: ("data_" + str(x), x))
    return interface

@pytest.fixture
def mock_shard_interface():
    interface = MagicMock()
    interface.host_to_accelerator = MagicMock(side_effect=lambda x: "processed_" + str(x))
    return interface

@pytest.fixture
def sharded_downloader(dist_manager_factory, mock_worker_interface, mock_shard_interface):
    dist_manager = dist_manager_factory(1)
    downloader = sdl.ShardedDataDownloader(
        worker_interface=mock_worker_interface,
        shard_interface=mock_shard_interface,
        dist_manager=dist_manager,
        workers_per_node=2,  # Assuming 2 workers per node
        batch_size=64,  # Total batch size
        queue_depth=10  # Queue depth
    )
    return downloader

@pytest.fixture
def sharded_downloaders(dist_manager_factory, mock_worker_interface, mock_shard_interface):
    downloaders = []
    for i in range(4):  # This will create 4 downloaders with PIDs 0 to 3
        dist_manager = dist_manager_factory(i)
        downloader = sdl.ShardedDataDownloader(
            worker_interface=mock_worker_interface,
            shard_interface=mock_shard_interface,
            dist_manager=dist_manager,
            workers_per_node=2,
            batch_size=64,
            queue_depth=10
        )
        downloaders.append(downloader)  # Correctly appending to list
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

    # Perform multiple step operations to validate round robin index handling
    num_steps = 10
    last_index = None
    for _ in range(num_steps):
        sharded_downloader.step()
        sharded_downloader.ack()
        # Assuming `step` updates `round_robin_index` correctly
        if last_index is not None:
            expected_index = 0 
            print(expected_index, sharded_downloader.round_robin_index)
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
        downloader.start(0)  # Start each downloader

    # Collect all batches from each downloader
    all_batches = []
    print("A",len(sharded_downloaders))
    for downloader in sharded_downloaders:
        processed_batch = downloader.step()  # Assuming step returns the processed data batch
        print("B",processed_batch)
        all_batches.append(processed_batch)
    print(all_batches)

    # Check each batch
    for i, downloader in enumerate(sharded_downloaders):
        batch_size = downloader.batch_size
        expected_elements = set(range(downloader.dist_manager.pid, batch_size, downloader.dist_manager.nodes))
        processed_elements = set(int(data.split("_")[-1]) for data in all_batches[i])
        print(list(expected_elements))
        print(list(processed_elements))

        assert processed_elements == expected_elements, f"Downloader {i} elements mismatch: expected {expected_elements}, got {processed_elements}"

    # Stop each downloader
    for downloader in sharded_downloaders:
        downloader.stop()


# Additional tests can be written to simulate failure scenarios, e.g., worker fails to fetch data