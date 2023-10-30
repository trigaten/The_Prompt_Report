# test_pipeline.py
import pytest
import os
from prompt_systematic_review.pipeline import *
import random
import time
import hashlib


def hashString(bytes):
    return hashlib.sha256(bytes).hexdigest()

@pytest.fixture
def client():
    return Pipeline(token=os.environ['HF_AUTH_TOKEN'])

def test_login():
    testClient = Pipeline()
    assert testClient.is_logged_in() == False
    testClient.login(os.environ['HF_AUTH_TOKEN'])
    assert testClient.is_logged_in() == True

def test_get_all_files(client):
    assert len(client.get_all_files()) > 0

def test_get_all_data_files(client):
    assert len(client.get_all_data_files()) > 0
    assert all([x.endswith(".csv") for x in client.get_all_data_files()])

def test_read_from_file(client):
    assert len(client.read_from_file("test.csv")) > 0
    assert len(client.read_from_file("test.csv").columns) == 2
    assert client.read_from_file("test.csv")[" Age"].mean() == 21

def test_write_to_file(client):
    lenOfFiles = len(client.get_all_files())
    randString = random.randbytes(100) + str(time.time()).encode()
    randHash = hashString(randString)
    csvDict = {"test":[1,3], "test2":[2,4]}
    client.write_to_file(f"test_{randHash}.csv", pd.DataFrame(csvDict))
    df = client.read_from_file(f"test_{randHash}.csv") 
    assert df["test"].sum() == 4
    assert df["test2"].sum() == 6
    client.fs.delete(f"test_{randHash}.csv")
    assert len(client.get_all_files()) == lenOfFiles
    
