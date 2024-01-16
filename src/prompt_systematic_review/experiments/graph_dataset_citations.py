import matplotlib.pyplot as plt
from prompt_systematic_review.config_data import DataFolderPath
import os

# Dataset names and citation counts, from Semantic Scholar
datasets = [
    "GSM8K",
    "Search_QA",
    "MMLU",
    "AQUA-RAT",
    "BIG-bench",
    "TruthfulQA",
    "CommonsenseQA",
    "QASC",
    "WinoGrande",
    "BBH",
    "HellaSwag",
]
citations = [702, 390, 608, 335, 629, 385, 816, 215, 387, 227, 594]


def graph_dataset_citations():
    # Sort the datasets and citations together, in descending order
    combined_data = list(zip(datasets, citations))
    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
    sorted_datasets, sorted_citations = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_datasets, sorted_citations, color="blue")
    plt.xlabel("Dataset Name")
    plt.ylabel("Number of Citations")
    plt.title("Dataset Citations")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            DataFolderPath,
            "experiments_output" + os.sep + "graph_dataset_citations_output.png",
        )
    )


class Experiment:
    def run():
        graph_dataset_citations()
