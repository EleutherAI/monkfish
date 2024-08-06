import argparse
import fs
from fs.osfs import OSFS
from fs_gcsfs import GCSFS
from google.oauth2 import service_account
from google.cloud import storage

def copy_to_gcs(source_path, bucket_name, target_path, credentials_path):
    # Load credentials from the service account file
    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    # Create a Google Cloud Storage client
    client = storage.Client(credentials=credentials, project=credentials.project_id)

    # Create filesystem objects
    source_fs = OSFS(source_path)
    gcs_fs = GCSFS(bucket_name, client=client)

    # Ensure the target directory exists
    gcs_fs.makedirs(target_path, recreate=True)

    # Copy contents
    fs.copy.copy_dir(source_fs, '/', gcs_fs, target_path)

    print(f"Contents of '{source_path}' copied to 'gs://{bucket_name}/{target_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy local directory to Google Cloud Storage using pyfilesystem2")
    parser.add_argument("source_path", help="Path of the local source directory")
    parser.add_argument("bucket_name", help="Name of the GCS bucket")
    parser.add_argument("target_path", help="Path of the target directory in GCS")
    parser.add_argument("credentials_path", help="Path to the service account credentials JSON file")

    args = parser.parse_args()

    copy_to_gcs(args.source_path, args.bucket_name, args.target_path, args.credentials_path)

