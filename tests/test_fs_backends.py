import os
import pytest
import catfish.lvd.shrd_data_loader as sdl

@pytest.fixture(scope="module")
def gcs_fs():
    """Create a connection to the GCS bucket."""
    bucket_name = "lvd_test"  # Specify your bucket name here
    credentials_path = "service-account-key.json"  # Update the path to your service account JSON file
    return sdl.gcp_filesystem(bucket_name, credentials_path=credentials_path)


def test_gcp_filesystem_upload_and_download(gcs_fs):
    test_file_path = "test_data.txt"
    test_content = "This is a test file."
    remote_file_path = "test_folder/" + test_file_path

    # Ensure the directory exists
    if not gcs_fs.exists('test_folder'):
        gcs_fs.makedirs('test_folder')

    # Write test content to a file locally
    with open(test_file_path, 'w') as f:
        f.write(test_content)

    # Upload the file to GCS
    with gcs_fs.openbin(remote_file_path, 'w') as remote_file:
        with open(test_file_path, 'rb') as local_file:
            remote_file.write(local_file.read())

    # Download the file back from GCS
    downloaded_content = ""
    with gcs_fs.openbin(remote_file_path, 'r') as remote_file:
        downloaded_content = remote_file.read().decode('utf-8')

    # Assert the uploaded content matches the downloaded content
    assert test_content == downloaded_content, "Content mismatch between uploaded and downloaded files."

    # Cleanup: remove the test file from GCS
    gcs_fs.remove(remote_file_path)
    gcs_fs.removedir('test_folder')

    # Cleanup local file
    os.remove(test_file_path)


def test_os_filesystem_operations():
    test_directory = "/tmp/test_osfs"  # Using /tmp directory which is typically used for temporary files in Unix-like systems
    test_file_name = "testfile.txt"
    test_content = "Hello, this is a test file!"

    # Ensure the directory exists
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    # Initialize the local filesystem
    local_fs = sdl.os_filesystem(test_directory)

    try:
        # Create and write to the file
        with local_fs.open(test_file_name, 'w') as file:
            file.write(test_content)

        # Read from the file
        with local_fs.open(test_file_name, 'r') as file:
            content_read = file.read()
            assert content_read == test_content, "Content read does not match content written."

    finally:
        # Clean up: Remove the test file and directory
        if local_fs.exists(test_file_name):
            local_fs.remove(test_file_name)
        if local_fs.exists(test_directory):
            local_fs.removetree('/')  # Be cautious with removetree, '/' refers to the root path within the OSFS instance, not the actual system root

        local_fs.close()