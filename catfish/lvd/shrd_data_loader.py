import multiprocessing
import multiprocessing.queues
import fs

def gcp_filesystem():
    pass

def osfs():
    pass

class ShardedDataUploader:
    def __init__(self, worker_interface, shard_interface, counter, dist_manager, workers_per_node=1):
        pass

class ShardedDataDownoader:
    def __init__(self, worker_interface, shard_interface, dist_manager, 
                 workers_per_node=1, batch_size=32, queue_depth=5):
        assert batch_size % dist_manager.nodes == 0

        #Start workers
        self.workers_per_node = workers_per_node
        self.batch_size = batch_size
        self.pid = dist_manager.pid
        self.nodes = dist_manager.nodes
        self.worker_interface_generator = worker_interface
        self.shard_interface_generator = shard_interface

        self.shard_interface = self.shard_interface_generator()

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

            queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=self._worker, args=(
                    start_index, queue, self.stop_event
                )
            )

            self.queues.append(queue)
            self.workers.append(worker)
    
    def stop(self):
        self.stop_event.set() 

        for worker in self.workers:
            worker.join()
        
        self.queues = None
        self.stop_event = None

    def _worker(self, start_index, queue, stop_event):
        counter = start_index
        worker_interface = self.worker_interface_generator
        
        while(not stop_event.is_set()):
            example = worker_interface.get_example(counter)
            queue.put(example)
            counter += self.workers_per_node*self.nodes

    def step(self):
        assert self.processed

        round_robin_index = self.round_robin_index
        for _ in range(self.batch_size // self.nodes):

            sub_batch_components = []
            sub_batch_ids = []
            for queue_index, queue in enumerate(self.queues):
                sub_batch_data, sub_batch_id = self.queues[round_robin_index].get()
                sub_batch_components.append(sub_batch_data)
                sub_batch_ids.append(sub_batch_id)
            
                #Verify expected sub_batch_id
                assert sub_batch_id == self.counter + queue_index*self.nodes + self.pid
            
            round_robin_index += 1
            round_robin_index %= self.workers_per_node

            local_data, data_id = self.queues[queue_index].get()
            assert data_id == self.counter
            
        accelerator_data = self.shard_interface(local_data)
        
        self.processed = False

        return accelerator_data

    def ack(self):
        assert (not self.processed)
        self.counter += self.batch_size
        self.round_robin += self.batch_size // self.nodes
        self.round_robin %= self.workers_per_node
        self.processed = True

class DummyWorkerInterface:
    """Interface to Video data, folder is a dataset, 1 train example per file"""
    def __init__(self, queue, fs):
        pass
    
    def list_dir():
        #List all examples in folder
    
    def get_example(self, example_id, mode="contiguous_video"):
        #Download and preprocess 1 example
        pass
    
    def upload_example(self, path):
        pass

class DummyShardInterface:
    def host_to_accelerator():
        pass
    
    def accelerator_to_host():
        pass

class VideoShardInterface:
    def host_to_accelerator():
        pass
    
    def accelerator_to_host():
        pass

class VideoWorkerInterface:
    """Interface to Video data, folder is a dataset, 1 train example per file"""
    def __init__(self, queue, fs):
        pass
    
    def list_dir():
        #List all examples in folder
        pass
    
    def get_example(self, example_id, mode="contiguous_video"):
        #Download and preprocess 1 example
        pass
    
    def upload_example(self, example_id):
        pass
    
class LatentMediumInterface:
    def list_dir():
        #List all examples in folder


