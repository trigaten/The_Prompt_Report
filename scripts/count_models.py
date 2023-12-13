import requests
import json
import pandas as pd
from collections import Counter, defaultdict
from prompt_systematic_review.semantic_scholar_source import SemanticScholarSource

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

file_path = "./master_papers.csv"

arxiv_papers_df = pd.read_csv(file_path)

paper_ids = set(arxiv_papers_df["paperId"])
print(paper_ids)

model_counts = defaultdict(list)

for model in model_names:
    papers_dict_lst = SemanticScholarSource().bulkSearchPapers(model)

    for paper_dict in papers_dict_lst:
        paper_id = paper_dict["paperId"]
        paper_title = paper_dict["title"]
        if paper_id in paper_ids:
            model_counts[model].append(paper_title)

print(model_counts)

file_path = "../data/model_citation_counts.txt"

with open(file_path, "w") as f:
    for model, titles in model_counts.items():
        f.write(f"{model} ({len(titles)}): {titles}\n")
