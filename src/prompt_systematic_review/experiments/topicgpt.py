import pandas as pd
from prompt_systematic_review.utils.topic_gpt_utils import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from prompt_systematic_review.config_data import DataFolderPath
import subprocess


def run_topic_gpt():
    """
    Run the TopicGPT experiment to generate a topic distribution for the papers.

    This function sets up the necessary data and prompt files, runs the generation script,
    generates a tree structure from the generated topics, filters out topics below a certain threshold,
    and visualizes the topic distribution using a horizontal bar chart.

    :return: None
    :rtype: None
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 17})

    data_folder = os.path.join(DataFolderPath, "topic-gpt-data")
    prompt_folder = os.path.join(data_folder, "prompt")
    data_file = os.path.join(data_folder, "master_papers.jsonl")
    generation_prompt = os.path.join(prompt_folder, "generation_1.txt")
    seed_1 = os.path.join(prompt_folder, "seed_1.md")
    generation_out = os.path.join(data_folder, "generation_1_paper.jsonl")
    generation_topic = os.path.join(data_folder, "master_paper.md")

    subprocess.run(
        [
            "python",
            "generation_1.py",
            "--deployment_name",
            "gpt-4-1106-preview",
            "--max_tokens",
            "300",
            "--temperature",
            "0.0",
            "--top_p",
            "0.0",
            "--data",
            data_file,
            "--prompt_file",
            generation_prompt,
            "--seed_file",
            seed_1,
            "--out_file",
            generation_out,
            "--topic_file",
            generation_topic,
            "--verbose",
            "True",
        ]
    )

    tree, nodes = generate_tree(read_seed(os.path.join(data_folder, "master_paper.md")))
    print(tree_view(tree))

    topic_count = sum([node.count for node in tree.descendants])
    threshold = 5
    for node in tree.descendants:
        if node.count < threshold and node.lvl == 1:
            print(f"Removing {node.name} ({node.count} counts)")
            node.parent = None
            nodes.remove(node)

    topics = [node.name for node in nodes]
    counts = [node.count for node in nodes]
    sorted_topics, sorted_counts = zip(
        *sorted(
            [(t, c) for t, c in zip(topics, counts)], key=lambda x: x[1], reverse=True
        )
    )
    plt.figure(figsize=(10, 20))
    plt.barh(sorted_topics[::-1], sorted_counts[::-1], color="purple")
    plt.xlabel("Number of papers")
    plt.title("Topic distribution")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            DataFolderPath, "experiments_output" + os.sep + "topic_distribution.png"
        ),
        format="pdf",
        bbox_inches="tight",
    )


class Experiment:
    def run():
        run_topic_gpt()


if __name__ == "__main__":
    run_topic_gpt()
