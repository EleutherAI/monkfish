import argparse
import fs
from fs_gcsfs import GCSFS
from google.oauth2 import service_account
from google.cloud import storage

def create_gcs_folder(bucket_name, folder_path, credentials_path):
    # Load credentials from the service account file
    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    # Create a Google Cloud Storage client
    client = storage.Client(credentials=credentials, project=credentials.project_id)

    # Create a GCSFS instance
    gcs_fs = GCSFS(bucket_name, client=client)

    # Use makedirs to create the folder (and any necessary parent folders)
    gcs_fs.makedirs(folder_path, recreate=True)

    print(f"Folder '{folder_path}' created in bucket '{bucket_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a folder in Google Cloud Storage using pyfilesystem2")
    parser.add_argument("bucket_name", help="Name of the GCS bucket")
    parser.add_argument("folder_path", help="Path of the folder to create")
    parser.add_argument("credentials_path", help="Path to the service account credentials JSON file")

    args = parser.parse_args()

    create_gcs_folder(args.bucket_name, args.folder_path, args.credentials_path)

