import os
import csv
import argparse
import pandas as pd
from tika import parser as tika_parser
from tqdm import tqdm
from collections import Counter, defaultdict
from prompt_systematic_review.semantic_scholar_source import SemanticScholarSource

"""This script counts the number of papers in our dataset that mention each {model/dataset/framework}.
The script takes one arg, for the path location of the full paper dataset."""

# TODO improvement: use word embeddings to find similar tool names.
# for now, assume the common tool name will appear in the paper
def count_tool_mentions(input_folder_path: str, output_file_path: str, tool_lst: list):
    tool_counts = defaultdict(list)
    
    # Iterate through all files in the input folder, count tool mentions
    for filename in tqdm(os.listdir(input_folder_path)):
        try:
            if filename.endswith(".pdf"):
                file_path = os.path.join(input_folder_path, filename)

                parsed_pdf = tika_parser.from_file(file_path)
                data = parsed_pdf["content"]

                for tool in tool_lst:
                    if tool in data:
                        tool_counts[tool].append(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("tool_counts: ", tool_counts)   
        
    with open(output_file_path, "w") as f:
      fieldnames = ["tool_name", "count", "list_of_papers"]

      # Create a CSV writer object
      writer = csv.DictWriter(f, fieldnames=fieldnames)

      # Write headers to the CSV file
      writer.writeheader()

      # Write data rows to the CSV file
      for tool, titles in tool_counts.items():
          writer.writerow(
              {"tool_name": tool, "count": len(titles), "list_of_papers": titles}
          )

# script portion
masterpaperscsv_file_path = "./master_papers.csv"

# get all paper ids from our dataset
arxiv_papers_df = pd.read_csv(masterpaperscsv_file_path)
paper_ids = set(arxiv_papers_df["paperId"])

tool_counts = defaultdict(list)

parser = argparse.ArgumentParser(description="Count tool mentions in paper dataset.")
parser.add_argument("file_path", type=str, help="Path of full paper dataset.")
args = parser.parse_args()
papers_dataset_path = args.file_path

# 1. call function to count model mentions
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
count_tool_mentions(papers_dataset_path, "../data/model_citation_counts.csv", model_names)

# 2. call function to count dataset mentions
dataset_names = [
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
count_tool_mentions(papers_dataset_path, "../data/dataset_citation_counts.csv", dataset_names)


# 3. call function to count framework mentions
framework_names = [
    "ReAct",
    "FuzzLLM",
    "RA-DIT",
    "BLSP",
    "Divide-and-Prompt",
    "TagGPT",
    "Models-Vote Prompting",
    "PEARL",
    "PromptNER",
    "Rewrite-Retrieve-Read",
    "GDP-Zero",
    "Dynamic Prompting",
    "IDAS",
    "Program Distillation",
    "LLMSmith",
    "RLPrompt",
    "AutoHint",
    "RPO",
    "Mobile-Edge AIGX",
    "Automatic Prompt Optimization (APO)",
    "Optimization by PROmpting (OPRO)",
    "Dialogue-Comprised Policy-Gradient-Based Discrete Prompt Optimization (DP_2O)",
    "CARP",
    "S^3 System",
    "Thought Propagation",
    "Automatic Prompt Engineer (APE)",
    "Prompt Adaptation",
]
count_tool_mentions(papers_dataset_path, "../data/framework_citation_counts.csv", framework_names)
