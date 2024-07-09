ACCELERATOR_TYPE_TO_HOST_BOUNDS = {
    'v2-8': '1,1,1',
    'v2-32': '2,2,1',
    'v2-128': '4,4,1',
    'v2-256': '4,8,1',
    'v2-512': '8,8,1',
    'v3-8': '1,1,1',
    'v3-32': '2,2,1',
    'v3-128': '4,4,1',
    'v3-256': '4,8,1',
    'v3-512': '8,8,1',
    'v3-1024': '8,16,1',
    'v3-2048': '16,16,1'
}

TPU_TYPE_SIZE = lambda x : int(x.split("-")[1])
TPU_HOST_COUNT = lambda x : int(x.split("-")[1])//8
TPU_HEALTHY = {"state": "READY", "health": "HEALTHY"}
TPU_PREEMPTED = {"state": "PREEMPTED"}
TPU_DELTETING = {"state": "DELETING"}
TPU_TERMINATED = {"state": "TPU_TERMINATED"}
TPU_REPAIRING = {"state": "REPAIRING"}
