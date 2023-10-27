from datasets import load_dataset, concatenate_datasets

"""
Gets Huggingface dataset
dataset_name: The name of the dataset on Huggingface
user_name: The name under which the dataset has been created
Returns a Dataset object
"""
def get(dataset_name, user_name):
    return load_dataset(user_name + "/" + dataset_name)

"""
Replaces entire database in Huggingface
dataset_name: The name of the dataset on Huggingface
new_dataset: A Dataset object containing the new dataset
"""
def push(dataset_name, new_dataset):
    new_dataset.push_to_hub(dataset_name)

"""
Appends to current dataset in Huggingface (experimental)
user_name: The name under which the dataset has been created
dataset_name: The name of the dataset on Huggingface
new_dataset: A Dataset object containing the new dataset
"""
def append(user_name, dataset_name, new_dataset):
    
    #Try-except for empty dataset case
    try:
        #Assuming split 'train' as it seems to be the default split
        dataset = load_dataset(user_name + "/" + dataset_name, split="train")
        appended_dataset = concatenate_datasets([dataset,new_dataset])
        appended_dataset.push_to_hub(dataset_name)
    except:
        push(dataset_name, new_dataset)



