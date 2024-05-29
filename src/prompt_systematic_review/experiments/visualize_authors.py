"""
This script generates visualizations of the publication counts
for authors in the master_papers.csv dataset.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from prompt_systematic_review.config_data import DataFolderPath
import os


def visualize_authors():
    file_path = os.path.join(DataFolderPath, "master_papers.csv")

    # Read the CSV file into a DataFrame
    arxiv_papers_df = pd.read_csv(file_path)

    # Split authors into a list and explode them to have one row per author
    authors = (
        arxiv_papers_df["authors"]
        .str.replace("'", "")
        .str.replace("]", "")
        .str.replace("[", "")
        .str.split(",")
        .explode()
    )

    # Count the occurrences of each author
    author_counts = authors.value_counts()

    # make a vertical bar chart displaying top 20 authors publication counts
    sorted_author_counts = author_counts.sort_values(ascending=False)
    top_20_authors = sorted_author_counts.head(20)

    plt.figure(figsize=(15, 8))
    top_20_authors.plot(kind="bar", rot=30, color="blue", edgecolor="black")
    plt.title("Publication Frequency by Author (Top 20)")
    plt.xlabel("Author")
    plt.ylabel("Number of Papers Included in Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(range(0, max(top_20_authors) + 2, 2))
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            DataFolderPath,
            "experiments_output" + os.sep + "publication_frequency_by_author.pdf",
        ),
        format="pdf",
        bbox_inches="tight",
    )

    # Display the frequency table
    publication_counts = author_counts.values
    frequency_table = pd.Series(publication_counts).value_counts().reset_index()
    frequency_table.columns = ["Publication Count", "Frequency"]
    frequency_table = frequency_table.sort_values(by="Publication Count")
    print(frequency_table)

    # make a histogram of the number of publications per author
    plt.figure(figsize=(10, 6))
    plt.bar(
        frequency_table["Publication Count"],
        frequency_table["Frequency"],
        color="blue",
        edgecolor="black",
    )
    plt.yscale("log")  # Logarithmic scale

    plt.xticks(frequency_table["Publication Count"])

    plt.xlabel("Publication Count")
    plt.ylabel("Frequency (log scale)")
    plt.title("Publication Count vs Frequency (Log Scale)")
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.savefig(
        os.path.join(
            DataFolderPath,
            "experiments_output" + os.sep + "publication_count_vs_frequency.pdf",
        ),
        format="pdf",
        bbox_inches="tight",
    )


class Experiment:
    def run():
        visualize_authors()
