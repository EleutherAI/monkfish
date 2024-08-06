from enum import Enum, unique, auto

import traceback
import threading
import contextvars
import ray
import monkfish.tpu.ray_tpu as rtpu
import random
import string
import monkfish.tpu.utils as u
import monkfish.tpu.tpu_constants as tc
import monkfish.tpu.tpu_host_actor as tha


def init():
    head_info = ray.init(include_dashboard=False)
    ip = ray.util.get_node_ip_address()
    address = f"{ip}:6379"
    return head_info, address

def shutdown():
    ray.shutdown()

@unique
class Dist(Enum):
    """
        SHARD_NODE: Distribute the object amongst the cpu nodes of a TPU
            The object must be indexable with length equal to the number of TPU nodes

        SHARD_CORE: Distribute the object amongst the cpu nodes of a TPU
            The object must be indexable with length equal to the number of TPU cores
            and it needs to be either a jax array or convertible to a jax array
        
        REPLICA_NODE: Distribute the object amongst the cpu nodes of a TPU
            There are no restrictions on this object

        REPLICA_CORE: Distribute the object amongst the cpu nodes of a TPU
            The object needs to be either a jax array or convertible to a jax array
    """
    SHARD_NODE = auto()
    SHARD_CORE = auto()
    REPLICA_NODE = auto()
    REPLICA_CORE = auto()


class TPUSwarm:
    """Represents a swarm of TPU clusters"""
    def __init__(tpu_clusters):
        pass

class ClusterConfig(object):
    def __init__(self,
            name, 
            tpu_type, 
            zone, 
            preemptible, 
            ssh_key_path, 
            head_info):
        self.name = name
        self.tpu_type = tpu_type
        self.zone = zone
        self.preemptible = preemptible
        self.ssh_key_path = ssh_key_path
        self.head_info = head_info
    
class TPUCluster(object):
    """Represents a single cluster of TPUs"""
    def __init__(self, name, tpu_type, zone, preemptible, ssh_key_path, head_info, address, owner=False, timeout=500):
        """setup tpu"""
        self.name = name
        self.tpu_type = tpu_type
        self.zone = zone
        self.preemptible = preemptible
        self.ssh_key_path = ssh_key_path
        self.head_info = head_info
        self.address = address
        self.owner=owner
        self.timeout=timeout
        
        self.cluster_id = u.gen_id()

        self.n_nodes = tc.TPU_HOST_COUNT(tpu_type)
        
        self.host_actors = [None]*self.n_nodes
        self.tpu_objects = set()

        if self.owner:
            print("making tpu...")
            self._create_tpu()
        print("establishing connection...")
        self._connect_to_tpu()
        print("starting actors...")
        self._start_actors()
        print("Initialization Complete!")

    def _create_tpu(self):
        result = rtpu.create_tpu(self.name,self.zone,self.tpu_type,self.preemptible)
        if (not result) or (not rtpu.tpu_wait_up(self.name, self.zone)):
            raise DeadTPUException("failed to create tpu")
       
    def _connect_to_tpu(self):
        try:
            #TODO:Replace with func-timeout
            with u.timeout(self.timeout):
                self.connections = rtpu.get_connections(
                        self.name, self.zone, self.ssh_key_path)
                self.n_hosts = len(self.connections)
                self.host_ids = [u.gen_id() for _ in range(self.n_hosts)]
                
                threads = []
                for i,conn in enumerate(self.connections):
                    print(f"setting up connection {i}...");
                    thread = threading.Thread(target=rtpu.setup_cluster, args=(conn,), daemon=True)
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()
                print("All connections started")

        except:
            print("UWU")
            raise DeadTPUException("failed to connect to tpu")


    def _start_actors(self):
        try:
            #start ray on the worker nodes
            #TODO:Paralellize
            for i in range(self.n_nodes):
                host_id = self.host_ids[i]
                rtpu.start_ray(
                        conn=self.connections[i],
                        address=self.address, 
                        host_id=host_id)
                #place actor on a node based on the host_id 
                pre_actor = tha.TPUActor.options(resources={host_id:1})
                self.host_actors[i] = pre_actor.remote()
        except:
            raise DeadTPUException("failed to start actors")
    
    def _disconnect_tpu(self):
        self.host_actors = [None]*self.n_nodes
        self.tpu_objects = set()
        
        #kill all actors and connections
        for conn in self.connections:
            rtpu.stop_ray(conn)
            conn.close()

    def _stop_tpu(self):
        #kill tpu cluster
        if self.owner:
            rtpu.delete_tpu(self.name, self.zone)

        assert rtpu.tpu_wait_down(self.name, self.zone)
        
    
    def __del__(self):
        """destroy tpu and clean up"""
        try:
            self._disconnect_tpu()
            if self.owner:
                self._stop_tpu()
        except:
            pass
    
    def disconnect(self):
        try:
            self._disconnect_tpu()
        except:
            pass

    def restart(self):
        while True:
            try:
                self._stop_tpu()
                if self.owner:
                    self._create_tpu()
                self._connect_to_tpu()
                self._start_actors()
                break
            except DeadTPUException as e:
                traceback.print_exc(e)
                pass
    
    def is_up(self):
        """return true if TPU is up"""
        return rtpu.tpu_up(self.name, self.zone)

    def __call__(self, object_handles):
        """tbd"""
        pass

    def put(self, values):
        """Sends the values to the TPU hosts"""
        try:
            handles = []
            for i,value in enumerate(values):
                handle = tha.ObjectHandle()
                self.host_actors[i].__setitem__.remote(handle,value)
                handles.append(handle)
            return TPUObject(handles,self)
        except ray.exceptions.RayActorError:
            raise DeadTPUException("tpu failed while adding value to tpu")
    
    def get(self, tpu_object):
        """Get values from a TPU host"""
        try:
            futures = []
            for i,handle in enumerate(tpu_object.object_handles):
                futures.append(self.host_actors[i].__getitem__.remote(handle))

            values = ray.get(futures)
            return values 
        except ray.exceptions.RayActorError:
            raise DeadTPUException("tpu failed while fetching result")

class TPUObject():
    """Wrapper to represent an object on a TPU Cluster"""
    def __init__(self, object_handles, tpu_cluster):
        self.object_handles = object_handles
        self.tpu_cluster = tpu_cluster
        #Use ray to distribute x to all remote workers
        self.id = u.gen_id()
        self.tpu_cluster.tpu_objects.add(self)

    def __call__(self, *args, **kwargs):
        try:
            actors = self.tpu_cluster.host_actors
            futures = []
            for i, actor in enumerate(actors):
                _f = self.object_handles[i]
                _args = [x.object_handles[i] for x in args]
                _kwargs = {key:x.object_handles[i] for key,x in kwargs.items()}
                _output = actor.__call__.remote(_f,*_args,**_kwargs)
                futures.append(_output)
            
            handles = ray.get(futures)
            return TPUObject(handles,self.tpu_cluster)
        except ray.exceptions.RayActorError:
            raise DeadTPUException("tpu failed during function call")

    def __hash__(self):
        return self.id.__hash__()
    
    def __eq__(self):
        #TODO
        pass

    def __del__(self):
        #TODO: Delete shit properly
        try:
            self.tpu_cluster.tpu_objects.remove(self)
        except:
            pass

class DeadTPUException(Exception):
    pass
