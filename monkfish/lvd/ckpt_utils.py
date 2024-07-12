import fs

import monkfish.lvd.fs_utils as fs_utils

def get_checkpoint_fs(ckpt_fs_type, ckpt_root_directory):
    ckpt_conf = config["transformer_ardm"]["checkpoints"]
    ckpt_fs_type = ckpt_conf["fs_type"]
    ckpt_root_directory = ckpt_conf["ckpt_root_directory"]
    
    fs_args = {
        'fs_type': ckpt_fs_type,
        'root_path': ckpt_root_directory
    }
    
    if ckpt_fs_type == "gcp":
        gcp_conf = config["gcp"]
        fs_args.update({
            'bucket_name': gcp_conf["gcp_bucket_name"],
            'credentials_path': gcp_conf["gcp_credentials_path"]
        })
    
    try:
        return fs_utils.fs_initializer(fs_args)
    except ValueError as e:
        raise ValueError(f"Error initializing filesystem: {str(e)}")

def empty_checkpoint_fs(ckpt_fs):
    def remove_recursive(path):
        for item in ckpt_fs.scandir(path):
            item_path = f"{path}/{item.name}"
            if item.is_dir:
                remove_recursive(item_path)
                ckpt_fs.removedir(item_path)
            else:
                ckpt_fs.remove(item_path)

    remove_recursive('/')
    print("Checkpoint filesystem has been completely emptied.")