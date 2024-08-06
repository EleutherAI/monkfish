import pytest
import monkfish.tpu as tpu
import monkfish.tpu.infrastructure as tpu_infra
import time
import os
import jax
import jax.numpy as jnp
import numpy as np

import monkfish.lvd.shrd_data_loader as dl
import monkfish.lvd.models.dist_utils as du

@pytest.fixture(scope="module")
def tpu_cluster():
    ssh_key_path = os.path.expanduser("~/.ssh/google_compute_engine")
    tpu_name = "greenland"
    tpu_size = "v3-32"
    region = "europe-west4-a"
    preemptible = True

    head_info, address = tpu_infra.init()
    print(head_info)

    greenland = tpu_infra.TPUCluster(tpu_name, tpu_size, region, 
            preemptible, ssh_key_path, head_info, address, owner=False)
    
    yield greenland
    
    greenland.disconnect()
    tpu_infra.shutdown()

def pmap_psum_test():
    def sum_and_double(x):
        return 2 * jax.lax.psum(x, axis_name='i')

    n_devices = jax.device_count()
    x = jnp.arange(n_devices)
    result = jax.pmap(sum_and_double, axis_name='i')(x)
    return result

def jax_test():
    print("A")
    jax.distributed.initialize()
    print("B")
    device_count = jax.device_count()
    local_device_count = jax.local_device_count()

    xs = jax.numpy.ones(jax.local_device_count())
    r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

    print('jax global device count:', jax.device_count())
    print('jax local device count:', jax.local_device_count())
    return r

def test_tpu_functionality(tpu_cluster):
    n = 4
    f = tpu_cluster.put([jax_test]*n)
    x = tpu_cluster.put([(x,n,8) for x in range(n)])
    steps = 1
    
    t1 = time.time()
    for i in range(steps):
        try:
            print("bla")
            r = None
            r = f()
            results = tpu_cluster.get(r)
            assert results is not None, "Result should not be None"
            #assert isinstance(result, jax.Array), "Result should be a numpy array"
            assert len(results) == 4, "Unexpected number of results"
            for result in results:
                assert result.shape == (8,), f"Unexpected shape: {result.shape}"
                assert np.allclose(result, 32), "Unexpected result values"
        except tpu_infra.DeadTPUException as e:
            print("TPU failure detected, restarting...")
            del f
            del r
            tpu_cluster.restart()
            f = tpu_cluster.put([jax_test]*n)
            x = tpu_cluster.put([(x,n,8) for x in range(n)])
    
    t2 = time.time()
    execution_time = (t2-t1)/n
    print(f"Average execution time per step: {execution_time:.6f} seconds")

def test_tpu_device_count(tpu_cluster):
    f = tpu_cluster.put([jax.device_count])
    result = tpu_cluster.get(f()[0])
    assert result == 32, f"Expected 32 devices for v3-32, but got {result}"

def test_tpu_local_device_count(tpu_cluster):
    f = tpu_cluster.put([jax.local_device_count])
    result = tpu_cluster.get(f()[0])
    assert result == 8, f"Expected 8 local devices, but got {result}"

def test_pmap_psum_communication(tpu_cluster):
    f = tpu_cluster.put([pmap_psum_test])
    try:
        result = tpu_cluster.get(f()[0])
        
        expected_sum = jax.device_count() * (jax.device_count() - 1) // 2
        expected_result = 2 * expected_sum
        
        assert result.shape == (jax.device_count(),), f"Unexpected shape: {result.shape}"
        assert np.all(result == expected_result), f"Expected all values to be {expected_result}, but got {result}"
        
        print(f"PMAP/PSUM test successful. Result: {result}")
    except tpu_infra.DeadTPUException as e:
        pytest.fail(f"TPU failure during PMAP/PSUM test: {str(e)}")

def test_latent_shard_interface_host_to_accelerator(tpu_cluster):
    def host_to_accelerator_test():
        print("A")
        jax.distributed.initialize()
        print("B")

        # Set up the mesh for the DistManager
        mesh_shape = (32, 1, 1)  # Adjust based on your TPU configuration
        
        dist_manager = du.DistManager(mesh_shape, None)
        
        # Create LatentShardInterface
        shard_interface = dl.LatentShardInterface(dist_manager)
        
        # Generate different data for each host
        local_device_count = jax.local_device_count()
        host_id = jax.process_index()
        local_batch_size = 16  # Adjust as needed
        
        # Create token data (let's assume it's just integers for simplicity)
        local_tokens = np.arange(local_batch_size) + host_id * 100
        
        # Create array data
        local_arrays = np.ones((local_batch_size, 16, 16)) * (host_id + 1)  # 16x16 arrays filled with host_id + 1
        
        # Prepare data in the format expected by host_to_accelerator
        local_data = [((tokens, arrays),) for tokens, arrays in zip(local_tokens, local_arrays)]
        
        # Call host_to_accelerator
        global_batch_size = local_batch_size * jax.process_count()
        global_data = shard_interface.host_to_accelerator(local_data, global_batch_size)
        
        # Verify the result
        global_tokens, global_arrays = global_data
        
        # Check shapes
        assert global_tokens.shape == (global_batch_size,), f"Expected shape {(global_batch_size,)}, got {global_tokens.shape}"
        assert global_arrays.shape == (global_batch_size, 16, 16), f"Expected shape {(global_batch_size, 16, 16)}, got {global_arrays.shape}"
        
        # Check distribution
        assert global_tokens.sharding.is_fully_replicated == False, "Tokens should be sharded"
        assert global_arrays.sharding.is_fully_replicated == False, "Arrays should be sharded"
        
        """
        # Gather data back to host for verification
        host_tokens = jax.device_get(global_tokens)
        host_arrays = jax.device_get(global_arrays)
        
        # Verify token values
        expected_tokens = np.concatenate([np.arange(local_batch_size) + i * 100 for i in range(jax.process_count())])
        assert np.array_equal(host_tokens, expected_tokens), f"Token mismatch. Expected {expected_tokens}, got {host_tokens}"
        
        # Verify array values
        for i in range(jax.process_count()):
            start = i * local_batch_size
            end = (i + 1) * local_batch_size
            assert np.all(host_arrays[start:end] == i + 1), f"Array mismatch for host {i}"
        """
        
        print("LatentShardInterface host_to_accelerator test passed successfully!")
    
    # Run the test on the TPU
    f = tpu_cluster.put([host_to_accelerator_test]*4)
    r = f()
    tpu_cluster.get(r)