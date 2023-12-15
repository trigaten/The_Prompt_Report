import os
import json
import argparse
import requests
import pandas as pd
from tika import parser as tika_parser
from tqdm import tqdm
from collections import Counter, defaultdict
from prompt_systematic_review.semantic_scholar_source import SemanticScholarSource

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


# script portion
masterpaperscsv_file_path = "./master_papers.csv"

# get all paper ids from our dataset
arxiv_papers_df = pd.read_csv(masterpaperscsv_file_path)
paper_ids = set(arxiv_papers_df["paperId"])

model_counts = defaultdict(list)

parser = argparse.ArgumentParser(description="Count model mentions in paper dataset.")
parser.add_argument("file_path", type=str, help="Path of full paper dataset.")
args = parser.parse_args()
papers_dataset_path = args.file_path

# call function to count model mentions
count_model_mentions(papers_dataset_path)
print(model_counts)

output_file_path = "../data/model_citation_counts.txt"

with open(output_file_path, "w") as f:
    for model, titles in model_counts.items():
        f.write(f"{model} ({len(titles)}): {titles}\n")
