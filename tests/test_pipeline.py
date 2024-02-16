# test_pipeline.py
import pytest
import os
from prompt_systematic_review.utils.pipeline import *
from huggingface_hub import delete_file
import random
import time
import hashlib
import shutil


def hashString(bytes):
    return str(hashlib.md5(bytes).hexdigest())


@pytest.fixture
def client():
    return Pipeline(revision="test")


@pytest.mark.API_test
def test_get_root_files(client):
    assert len(client.get_root_files()) > 0


@pytest.mark.API_test
def test_read_from_file(client):
    assert len(client.read_from_file("test.csv")) > 0
    assert len(client.read_from_file("test.csv").columns) == 2
    assert client.read_from_file("test.csv")["Age"].mean() == 21


@pytest.mark.API_test
def test_write_to_file(client):
    lenOfFiles = len(client.get_root_files())
    randString = random.randbytes(100) + str(time.time()).encode()
    randHash = hashString(randString)
    csvDict = {"test": [1, 3], "test2": [2, 4]}
    print(client.revision)
    client.write_to_file(f"{randHash[:10]}_test.csv", pd.DataFrame(csvDict))
    print(client.revision)
    time.sleep(1)
    # assert client.revision == "main"
    df = client.read_from_file(f"{randHash[:10]}_test.csv")
    assert df["test"].sum() == 4
    assert df["test2"].sum() == 6
    # time.sleep(1)
    print(client.root + f"{randHash[:10]}_test.csv")
    delete_file(
        f"{randHash[:10]}_test.csv",
        "PromptSystematicReview/Prompt_Systematic_Review_Dataset",
        repo_type="dataset",
        revision="test",
    )

    assert len(client.get_root_files()) == lenOfFiles


@pytest.mark.API_test
def test_upload_file(client):
    # create and populate "test.txt"
    with open("test.txt", "w") as file:
        file.write("Test file")
    # Upload a file
    client.upload_file("test.txt")
    # Check if the file exists in the root files
    assert "test.txt" in [os.path.basename(i) for i in client.get_root_files()]

    # Clean up: Delete the uploaded file
    client.delete_file("test.txt")
    print([os.path.basename(i) for i in client.get_root_files()])
    assert "test.txt" not in [os.path.basename(i) for i in client.get_root_files()]
    os.remove("test.txt")


@pytest.mark.API_test
def test_download_file(client):
    # create and populate "test.txt"
    with open("test.txt", "w") as file:
        file.write("Test file")
    # Upload a file
    client.upload_file("test.txt")

    os.makedirs("./downloads", exist_ok=True)
    # Download the file
    client.download_file("test.txt", downloadPath="./downloads")

    # Check if the file exists in the download path
    assert os.path.exists("./downloads/test.txt")

    # Clean up: Delete the uploaded file and the downloaded file
    client.delete_file("test.txt")
    os.remove("./downloads/test.txt")
    os.removedirs("./downloads")


@pytest.mark.API_test
def test_delete_file(client):
    # Upload a file
    with open("test.txt", "w") as file:
        file.write("Test file")
    client.upload_file("test.txt")

    # Check if the file exists in the root files
    assert "test.txt" in [os.path.basename(i) for i in client.get_root_files()]

    # Delete the file
    client.delete_file("test.txt")

    # Check if the file is deleted
    assert "test.txt" not in [os.path.basename(i) for i in client.get_root_files()]

    # cleanup
    os.remove("test.txt")


@pytest.mark.API_test
def test_upload_folder(client):
    # Create a temporary folder and files
    os.makedirs("./temp_folder", exist_ok=True)
    with open("./temp_folder/testfile1.txt", "w") as file:
        file.write("Test file 1")
    with open("./temp_folder/testfile2.txt", "w") as file:
        file.write("Test file 2")

    # Upload the folder
    client.upload_folder("./temp_folder", folderInRepo="temp_folder")

    # Check if the folder exists in the root files
    assert "testfile1.txt" in [os.path.basename(i) for i in client.get_root_files()]
    assert "testfile2.txt" in [os.path.basename(i) for i in client.get_root_files()]

    # Clean up: Delete the uploaded folder and files
    client.delete_folder("temp_folder")
    assert "testfile1.txt" not in [os.path.basename(i) for i in client.get_root_files()]
    assert "testfile2.txt" not in [os.path.basename(i) for i in client.get_root_files()]

    os.remove("./temp_folder/testfile1.txt")
    os.remove("./temp_folder/testfile2.txt")
    # delete ./temp_folder even if it is not empty
    shutil.rmtree("./temp_folder")


@pytest.mark.API_test
def test_download_dataset(client):
    # Create a temporary folder for downloading the dataset
    os.makedirs("./downloads", exist_ok=True)

    # Download the dataset
    client.download_dataset(downloadPath="./downloads")

    # Check if the dataset file exists in the download path
    assert len(os.listdir("./downloads")) > 0

    # Clean up: Delete the downloaded dataset file
    shutil.rmtree("./downloads")
