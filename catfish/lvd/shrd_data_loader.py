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

class ShardedDataDownloader:
    def __init__(self, worker_interface, shard_interface, dist_manager, 
                 workers_per_node=1, batch_size=32, queue_depth=5):
        assert batch_size % dist_manager.nodes == 0

        #Start workers
        self.workers_per_node = workers_per_node
        self.batch_size = batch_size
        self.queue_depth = queue_depth
        self.dist_manager = dist_manager
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

            queue = multiprocessing.Queue(maxsize=self.queue_depth)
            worker = multiprocessing.Process(
                target=self._worker, args=(
                    start_index, queue, self.stop_event
                )
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
                except queue.Empty:
                    continue

        for worker in self.workers:
            worker.join()
        
        self.queues = None
        self.stop_event = None

    def _worker(self, start_index, queue, stop_event):
        counter = start_index
        worker_interface = self.worker_interface_generator
        
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

class DummyWorkerInterface:
    """Interface to Video data, folder is a dataset, 1 train example per file"""
    def __init__(self, queue, fs):
        pass
    
    def list_dir():
        #List all examples in folder
        pass
    
    def get_example(self, example_id):
        #Download and preprocess 1 example
        return None, example_id
    
    def upload_example(self, example_id):
        pass


class VideoWorkerInterface:
    """Interface to Video data, folder is a dataset, 1 train example per file"""
    def __init__(self, queue, fs):
        pass
    
    def list_dir():
        #List all examples in folder
        pass
    
    def get_example(self, example_id):
        #Download and preprocess 1 example
        pass
    
    def upload_example(self, example_id):
        pass

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


