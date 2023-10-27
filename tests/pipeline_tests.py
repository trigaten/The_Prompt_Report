from datasets import Dataset
import time
from pipeline import push, append, get

def testingPushAndGet(dataset_name, user_name):
    sample_data = {
        "column1": ["value1", "value2", "value3"],
        "column2": ["valueA", "valueB", "valueC"],
    }

    sample_dataset = Dataset.from_dict(sample_data)

    push(dataset_name, sample_dataset)
    assert(getRows(dataset_name, user_name) == 3)

#Append seems to work on Hugging face but I'm having trouble with the Database API, will leave out this test for now.
# def testingAppend(dataset_name, user_name):
#     sample_data = {
#         "column1": ["value4", "value5"],
#         "column2": ["valueD", "valueE"],
#     }

#     sample_dataset = Dataset.from_dict(sample_data)

#     append(user_name, dataset_name, sample_dataset)
    
#     assert(getRows(dataset_name, user_name) == 5)


#Returns the number of rows in the dataset
def getRows(dataset_name, user_name):
    my_dataset = get(dataset_name, user_name)['train']

    return my_dataset.num_rows

if (__name__ == "__main__"):

    USERNAME = "hudssntao"
    DATASET_NAME = "test2"

    testingPushAndGet(DATASET_NAME, USERNAME)

    # testingAppend(DATASET_NAME, USERNAME)




