import os
import csv
import json
import argparse
import requests
import pandas as pd
from tika import parser as tika_parser
from tqdm import tqdm
from collections import Counter, defaultdict
from prompt_systematic_review.get_papers.semantic_scholar_source import (
    SemanticScholarSource,
)
from prompt_systematic_review.config_data import DataFolderPath

"""This script counts the number of papers in our dataset that mention each model.
The script takes one arg, for the path location of the full paper dataset."""

# TODO improvement: use word embeddings to find similar model names.
# for now, assume the common model name will appear in the paper
model_names = [
    "GPT-3",
    "GPT-4",
    "InstructGPT",
    "Codex",
    "BLOOM",
    "BLOOMZ",
    "OPT",
    "LLaMA",
    "Codellama",
    "Lambda",
    "PaLM",
    "LLaVA",
    "CODEGEN",
    "SynthIE",
    "FLAN",
    "BERT",
    "RoBERTa",
    "BioBERT",
    "FinBERT",
    "GatorTron",
    "BART",
    "DreamFusion",
    "CLIP",
    "CoCoOp",
    "BLIP-2",
    "Vision Transformer",
    "Flamingo",
    "Grounding DINO",
    "YOLOv5m",
    "CLIPSeg",
    "VLP",
    "XMem",
    "SAM",
]


def count_models():
    # script portion
    masterpaperscsv_file_path = os.path.join(DataFolderPath, "master_papers.csv")

    # get all paper ids from our dataset
    arxiv_papers_df = pd.read_csv(masterpaperscsv_file_path)
    paper_ids = set(arxiv_papers_df["paperId"])

    model_counts = defaultdict(list)

    def count_model_mentions(folder_path):
        # Iterate through all files in the folder, count model mentions
        for filename in tqdm(os.listdir(folder_path)):
            try:
                if filename.endswith(".pdf"):
                    file_path = os.path.join(folder_path, filename)

                    parsed_pdf = tika_parser.from_file(file_path)
                    data = parsed_pdf["content"]

                    for model in model_names:
                        if model in data:
                            model_counts[model].append(filename)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    papers_dataset_path = os.path.join(DataFolderPath, "papers/")

    # call function to count model mentions
    count_model_mentions(papers_dataset_path)
    print(model_counts)

    output_file_path = os.path.join(DataFolderPath, "model_citation_counts.csv")

    with open(output_file_path, "w", encoding="utf-8") as f:
        fieldnames = ["model_name", "count", "list_of_papers"]

        # Create a CSV writer object
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write headers to the CSV file
        writer.writeheader()

        # Write data rows to the CSV file
        for model, titles in model_counts.items():
            writer.writerow(
                {"model_name": model, "count": len(titles), "list_of_papers": titles}
            )


class Experiment:
    def run():
        count_models()
