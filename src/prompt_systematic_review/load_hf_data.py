from datasets import load_dataset
from datasets import Split


def load_hf_dataset(dataset_name: str, name: str = None, split: str or Split = None):
    """
    Load in a Hugging Face dataset.

    :param dataset_name: The name of the Hugging Face dataset to load.
    :type dataset_name: str
    :param name: Defines the name of the dataset configuration
    :type name: str
    :param split: Which split of the data to load. If None, will return a dict with all splits (typically datasets.Split.TRAIN and datasets.Split.TEST).
    :type split: str or Split
    :return: The loaded dataset.
    :rtype: Dataset or DatasetDict
    """

    try:
        return load_dataset(
            dataset_name, name if name else None, split=split if split else None
        )
    except FileNotFoundError:
        print(
            f"The dataset {dataset_name} with config {name} and split {split} is not available on Hugging Face datasets."
        )
