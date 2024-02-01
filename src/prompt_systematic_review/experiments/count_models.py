import os
import csv
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pdfminer.high_level import extract_text
from tqdm import tqdm
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


def parse_pdf(file_path):
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def process_file(args):
    folder_path, filename = args
    file_path = os.path.join(folder_path, filename)
    if filename.endswith(".pdf"):
        data = parse_pdf(file_path)
        counts = {model: data.count(model) for model in model_names if model in data}
        return filename, counts
    return filename, {}

def count_model_mentions_parallel(folder_path):
    files = os.listdir(folder_path)
    with Pool(cpu_count()) as pool:
        # Use imap_unordered for better tqdm compatibility
        result_iter = pool.imap_unordered(process_file, [(folder_path, f) for f in files])
        
        model_counts = defaultdict(list)

        # Wrap the iterator with tqdm for the progress bar
        for filename, counts in tqdm(result_iter, total=len(files)):
            for model, count in counts.items():
                if count > 0:
                    model_counts[model].append(filename)

    return model_counts

def count_models():
    masterpaperscsv_file_path = os.path.join(DataFolderPath, "master_papers.csv")
    arxiv_papers_df = pd.read_csv(masterpaperscsv_file_path)
    paper_ids = set(arxiv_papers_df["paperId"])

    papers_dataset_path = os.path.join(DataFolderPath, "papers/")
    model_counts = count_model_mentions_parallel(papers_dataset_path)

    output_file_path = os.path.join(DataFolderPath, "model_citation_counts.csv")
    with open(output_file_path, "w", encoding="utf-8") as f:
        fieldnames = ["model_name", "count", "list_of_papers"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, titles in model_counts.items():
            writer.writerow(
                {"model_name": model, "count": len(titles), "list_of_papers": titles}
            )

class Experiment:
    def run():
        count_models()
