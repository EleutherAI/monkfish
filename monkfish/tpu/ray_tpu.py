import functools
import os
import subprocess
import time

import glob
import requests
import fabric as fa

import monkfish.tpu.tpu_constants as tc 

DEFAULT_POLL_PERIOD=10

def get_env_dict(tpu_type):
    pass

@functools.lru_cache()
def get_bearer():
    cmd = "gcloud auth print-access-token"
    output = subprocess.check_output(cmd, shell=True)
    return output.decode("utf-8").strip()

@functools.lru_cache()
def get_project():
    cmd = "gcloud config list --format 'value(core.project)'"
    output = subprocess.check_output(cmd, shell=True)
    return output.decode("utf-8").strip()

def create_tpu(name, zone, tpu_type, preemptible):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
        'Content-Type': 'application/json'
    }
    params = (('node_id', name),)
    data = {
        "accelerator_type": tpu_type,
        "runtime_version": "v2-alpha",
        "network_config": {"enable_external_ips": True}
    }

    if preemptible:
        data["schedulingConfig"] = {"preemptible": True}

    string = f'https://tpu.googleapis.com\
/v2alpha1/projects/{get_project()}/locations/{zone}/nodes'
    response = requests.post(string, headers=headers, params=params, json=data)
    return response.status_code == 200

def check_tpu(name, zone):
    headers = { 'Authorization': f'Bearer {get_bearer()}'}
    string = f'https://tpu.googleapis.com\
/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}'
    response = requests.get(string, headers=headers)
    return response.json()

def delete_tpu(name, zone):
    headers = {'Authorization': f'Bearer {get_bearer()}'}
    response = requests.delete(
        f'https://tpu.googleapis.com\
/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}',
        headers=headers)
    return response.json()

def tpu_match_state(tpu_info,state):
    matches = True
    for k, expected_v in state.items():
        if (k not in tpu_info) or (tpu_info[k] != expected_v):
            matches = False
            break
    return matches

def tpu_up(tpu_info):
    """Check if tpu is up and in a good state"""
    return tpu_match_state(tpu_info, tc.TPU_HEALTHY)

def tpu_wait_up(name, zone, poll_period=DEFAULT_POLL_PERIOD):
    """Wait for tpu to come online"""
    while True:
        time.sleep(poll_period)
        tpu_info = check_tpu(name, zone)
        if tpu_up(tpu_info):
            print("TPU is up")
            return True
        if "error" in tpu_info:
            print("Error detected while bringing up TPU")
            return False
        if tpu_match_state(tpu_info, tc.TPU_PREEMPTED):
            return False
        if tpu_match_state(tpu_info, tc.TPU_DELTETING):
            return False
        if tpu_match_state(tpu_info, tc.TPU_REPAIRING):
            return False

def tpu_wait_down(name, zone, poll_period=DEFAULT_POLL_PERIOD):
    while True:
        time.sleep(poll_period)
        tpu_info = check_tpu(name, zone)
        #Wait till TPU is gone
        if ("error" in tpu_info) and (tpu_info["error"]["code"] == 404):
            print("TPU is down")
            return True 


def get_connections(name, zone, key_path):
    info = check_tpu(name, zone)
    outputs = []

    #Ensure appropriate SSH key is present on all nodes
    command = ["gcloud","compute","tpus"
                ,"tpu-vm","ssh", name, f"--zone={zone}",
                f"--worker=all", "--command=:"]
    result = subprocess.run(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:  # Successful execution
        print("Command executed successfully:", result.stdout)
    else:  # There was an error
        print("Error:", result.stderr)

    for endpoint in info["networkEndpoints"]:
        print(endpoint["ipAddress"])
        outputs.append(
                fa.Connection(endpoint["ipAddress"],
                connect_kwargs={"key_filename": key_path}))
    return outputs

def setup_cluster(conn):
    """pip install both symphony and the local project"""
    #install locations
    if "PROJECT_SOURCE" in os.environ:
        project_source =  os.environ["PROJECT_SOURCE"]
    else: 
        raise NotImplementedError
    if "PROJECT_SOURCE_TYPE" in os.environ:
        project_source_type = os.environ["PROJECT_SOURCE_TYPE"]
    else: 
        project_source_type = "local"

    
    RUNTIME_ROOT = "/runtime"
    try:
        conn.sudo("rm -rf /runtime", hide=False)
    except:
        pass
    conn.sudo(f"mkdir {RUNTIME_ROOT}", hide=False)
    conn.sudo(f"ls {RUNTIME_ROOT}", hide=False)

    if project_source_type == "local":
        #add trailing slash if nessesary 
        fabric_copy(conn, project_source, 
                f"{RUNTIME_ROOT}/catfish")
        conn.sudo(f"{RUNTIME_ROOT}/catfish/scripts/configure_tpu.sh", hide=True)
    else:
        raise NotImplementedError

#TODO:Support directory mode settings 
def fabric_copy(conn, source, destination):
    conn.sudo(f"mkdir {destination}", hide=True)
    for root, dirs, files in os.walk(source):
        if "__pycache__" in root:
            continue
        for name in files:
            local_path = os.path.join(root, name)
            remote_path = swap_path(
                source, destination,local_path)
            sudo_put(conn,local_path, remote_path)

        for name in dirs:
            if name == "__pycache__":
                pass
            local_path = os.path.join(root, name)
            remote_path = swap_path(
                source, destination,local_path)
            conn.sudo(f"mkdir {remote_path}", hide=True)

def swap_path(old_prefix, new_prefix, full_path):
    rel_path = os.path.relpath(full_path, old_prefix)	
    new_path = os.path.join(new_prefix,rel_path)
    return new_path

def sudo_put(conn,local_path,remote_path):
    remote_root,remote_name = os.path.split(remote_path)
    conn.put(local_path)
    string = f"mv {remote_name} {remote_path}"
    conn.sudo(string, hide=True)

def start_ray(conn, address, host_id):
    """Start ray on the remote machine"""
    print("A:",address)
    resource_string = f"'{{\"{host_id}\": 1}}'"
    command = f"/runtime/catfish/scripts/start_ray.sh {address} {resource_string} 50000000000"
    
    # Execute the command
    conn.sudo(command)

def stop_ray(conn):
    #TODO: FIX
    try:
        conn.sudo('rm -rf /dev/shm', hide=True)
        conn.sudo('source ~/.bashrc && conda activate catfish && ray stop -f', hide=True)
    except:
        pass
