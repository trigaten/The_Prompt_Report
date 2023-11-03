# test_pipeline.py
import pytest
import os
from prompt_systematic_review.pipeline import *
from huggingface_hub import delete_file
import random
import time
import hashlib

'''
def hashString(bytes):
    return str(hashlib.md5(bytes).hexdigest())


@pytest.fixture
def client():
    return Pipeline(token=os.environ["HF_AUTH_TOKEN"], revision="test")


def test_login():
    testClient = Pipeline(revision="test")
    assert testClient.is_logged_in() == False
    testClient.login(os.environ["HF_AUTH_TOKEN"])
    assert testClient.is_logged_in() == True


def test_get_all_files(client):
    assert len(client.get_all_files()) > 0


def test_get_all_data_files(client):
    assert len(client.get_all_data_files()) > 0
    assert all([x.endswith(".csv") for x in client.get_all_data_files()])


def test_read_from_file(client):
    assert True
    assert len(client.read_from_file("test.csv")) > 0
    assert len(client.read_from_file("test.csv").columns) == 2
    assert client.read_from_file("test.csv")["Age"].mean() == 21


def test_write_to_file(client):
    assert True
    
    lenOfFiles = len(client.get_all_files())
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

    assert len(client.get_all_files()) == lenOfFiles
    '''
