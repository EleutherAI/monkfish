import pytest
import monkfish.tpu as tpu
import monkfish.tpu.infrastructure as tpu_infra
import time
import os
import jax
import numpy as np

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
    
    tpu_infra.shutdown()

def pmap_psum_test():
    def sum_and_double(x):
        return 2 * jax.lax.psum(x, axis_name='i')

    n_devices = jax.device_count()
    x = jnp.arange(n_devices)
    result = jax.pmap(sum_and_double, axis_name='i')(x)
    return result

def jax_test():
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
    steps = 5000
    
    t1 = time.time()
    for i in range(steps):
        try:
            r = f()
            result = tpu_cluster.get(r)
            assert result is not None, "Result should not be None"
            assert isinstance(result, np.ndarray), "Result should be a numpy array"
            assert result.shape == (jax.local_device_count(),), f"Unexpected shape: {result.shape}"
            assert np.allclose(result, jax.local_device_count()), "Unexpected result values"
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
    
    assert execution_time < 1.0, f"Execution time ({execution_time:.6f} s) exceeded threshold (1.0 s)"

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
