import monkfish.tpu as tpu
import monkfish.tpu.infrastructure as tpu_infra
import time
import os
import pickle

import jax

def main():
    print("hello world!")


    ssh_key_path = os.path.expanduser("~/.ssh/google_compute_engine")
    tpu_name = "greenland"
    tpu_size = "v3-32"
    region = "europe-west4-a"
    preemptible = True

    head_info, address = tpu_infra.init()
    print(head_info)

    greenland = tpu_infra.TPUCluster(tpu_name, tpu_size, region, 
            preemptible, ssh_key_path, head_info, address, owner=False)
    
    
    n = 4
    f = greenland.put([jax_test]*n)
    x = greenland.put([(x,n,8) for x in range(n)])
    steps = 500
    t1 = time.time()
    for i in range(steps):
        try:
            r = None
            r = f()
            greenland.get(r)
        except tpu_infra.DeadTPUException as e:
            print("TPU failure detected, restarting...")
            del f
            del r
            greenland.restart()
            f = greenland.put([jax_test]*n)
            x = greenland.put([(x,n,8) for x in range(n)])
            
    t2 = time.time()
    print((t2-t1)/n)

    greenland.disconnect()
    tpu_infra.shutdown()

def jax_test():

    # The total number of TPU cores in the pod
    device_count = jax.device_count()
    # The number of TPU cores attached to this host
    local_device_count = jax.local_device_count()

    # The psum is performed over all mapped devices across the pod
    xs = jax.numpy.ones(jax.local_device_count())
    r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

    # Print from a single host to avoid duplicated output
    print('jax global device count:', jax.device_count())
    print('jax local device count:', jax.local_device_count())
    #print('pmap result:', r)
    return r



if __name__ == "__main__":
    """Program entry point, runs on the head node"""
    main()
