import fs
import fs.osfs
import fs_gcsfs

import google.cloud
import google.oauth2
import time

def gcp_filesystem(bucket_name, root_path, credentials_path):
    time.sleep(3)
    """
    if not credentials_path:
        raise ValueError("Credentials path must be provided for authentication.")
    """

    # Load credentials from the service account file
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(credentials_path)

    # Initialize the GCS client with the provided credentials
    client = google.cloud.storage.Client(credentials=credentials, project=credentials.project_id)

    # Create a filesystem instance using the GCS client
    google_cloud_storage_fs = fs_gcsfs.GCSFS(bucket_name=bucket_name, root_path=root_path, client=client)
    return google_cloud_storage_fs

def os_filesystem(root_path):
    local_fs = fs.osfs.OSFS(root_path)
    return local_fs

def fs_initializer(args):
    fs_type = args.get('fs_type')

    if fs_type == 'gcp':
        bucket_name = args.get('bucket_name')
        root_path = args.get('root_path', '')
        credentials_path = args.get('credentials_path')

        if not bucket_name:
            raise ValueError("Bucket name must be provided for GCP filesystem.")
        if not credentials_path:
            raise ValueError("Credentials path must be provided for GCP filesystem.")

        return gcp_filesystem(bucket_name, root_path, credentials_path)

    elif fs_type == 'os':
        root_path = args.get('root_path')

        if not root_path:
            raise ValueError("Root path must be provided for OS filesystem.")

        return os_filesystem(root_path)

    else:
        raise ValueError(f"Unsupported filesystem type: {fs_type}")

def clear_and_remove_dir(filesystem, path):
    """Helper function to clear and remove a directory"""
    if path == '/':
        # Special handling for root directory
        for item in filesystem.scandir(path):
            full_path = fs.path.combine(path, item.name)
            if filesystem.isdir(full_path):
                filesystem.removetree(full_path)
            else:
                filesystem.remove(full_path)
        print(f"Cleared root directory: {path}")
    else:
        filesystem.removetree(path)
        print(f"Removed directory: {path}")