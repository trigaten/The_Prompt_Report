import os
import requests
import tarfile
import shutil


def download_and_extract(url, target_path):
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.raw.read())


def extract_tar(file_path, extract_path):
    # Extract the .tar file
    with tarfile.open(file_path) as tar:
        tar.extractall(path=extract_path)


def move_and_rename_extracted_contents(extracted_folder, final_folder, new_folder_name):
    # Move and rename the contents of the extracted folder
    mmlu_folder = os.path.join(final_folder, new_folder_name)
    os.makedirs(mmlu_folder, exist_ok=True)

    for item in os.listdir(extracted_folder):
        item_path = os.path.join(extracted_folder, item)
        shutil.move(item_path, mmlu_folder)

    return mmlu_folder


# URL of the .tar file
url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

# Temporary paths
download_path = "./data.tar"
extract_path = "./extracted"

# Final path
final_data_folder = "./data"
final_folder_name = "mmlu"

# Download and extract the file
download_and_extract(url, download_path)
extract_tar(download_path, extract_path)

# Move and rename the contents of the extracted folder
move_and_rename_extracted_contents(extract_path, final_data_folder, final_folder_name)

# Cleanup
if os.path.exists(download_path):
    os.remove(download_path)
if os.path.exists(extract_path):
    shutil.rmtree(extract_path)
