import os
import requests
import time
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import csv
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
from prompt_systematic_review.config_data import DataFolderPath


class SemanticScholarAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def rate_limited_request(self, url, headers, max_retries=3, delay=0.1):
        retries = 0
        while retries < max_retries:
            try:
                time.sleep(delay)
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if response.status_code in [429, 504]:
                    print(f"Received status code {response.status_code}. Retrying...")
                    delay *= 2  # Exponential backoff
                    retries += 1
                else:
                    print(f"HTTPError: {e}")
                    break
            except requests.RequestException as e:
                print(f"Request error: {e}")
                break
        return None

    def get_references(self, paper_id):
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,authors&limit=1000"
        headers = {"x-api-key": self.api_key}
        response = self.rate_limited_request(url, headers)
        if response and response.status_code == 200:
            data = response.json()
            reference_ids = [
                ref["citedPaper"]["paperId"]
                for ref in data["data"]
                if "citedPaper" in ref
            ]
            return reference_ids
        return []


class PaperProcessor:
    def __init__(self, api_key):
        self.semantic_scholar_api = SemanticScholarAPI(api_key)

    def process_papers(self, csv_file_path):
        paper_references = {}
        if os.path.exists(csv_file_path):
            with open(csv_file_path, mode="r", encoding="utf-8") as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=",")
                all_papers = list(csv_reader)
                for row in tqdm(all_papers, desc="Processing Papers"):
                    paper_id = row.get("paperId", "").strip()
                    references = self.semantic_scholar_api.get_references(paper_id)
                    paper_references[paper_id] = references
        else:
            print(f"CSV file does not exist: {csv_file_path}")
        return paper_references


class GraphVisualizer:
    def visualize_citation_counts(self, paper_references, technique_to_title):
        citation_counts = {tech: 0 for tech in technique_to_title.values()}
        for paper_id, refs in paper_references.items():
            for ref_id in refs:
                if ref_id in paper_references:
                    citation_counts[technique_to_title.get(ref_id, "Unknown")] += 1
        techniques, counts = zip(*citation_counts.items())
        plt.figure(figsize=(10, 5))
        plt.bar(techniques, counts, color="blue")
        plt.xlabel("Techniques")
        plt.ylabel("Citation Counts")
        plt.title("Citation Counts by Technique")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                DataFolderPath, "experiments_output" + os.sep + "paper_graph.pdf"
            ),format="pdf", bbox_inches="tight"
        )


class Main:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.paper_processor = PaperProcessor(self.api_key)
        self.graph_visualizer = GraphVisualizer()

    def run(self, csv_file_path, technique_to_title):
        paper_references = self.paper_processor.process_papers(csv_file_path)
        self.graph_visualizer.visualize_citation_counts(
            paper_references, technique_to_title
        )


if __name__ == "__main__":
    main = Main()
    titles = [
        "Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints",
        "Language Models are Few-Shot Learners",
        "A Survey on In-context Learning",
        "What Makes Good In-Context Examples for GPT-3?",
        "Finding Support Examples for In-Context Learning",
        "Unified Demonstration Retriever for In-Context Learning",
        "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity",
        "Reordering Examples Helps during Priming-based Few-Shot Learning",
        "Learning To Retrieve Prompts for In-Context Learning",
        "Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator",
        "Large Language Models are Zero-Shot Reasoners",
        "Large Language Models Are Human-Level Prompt Engineers",
        "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models",
        "Thread of Thought Unraveling Chaotic Contexts",
        "When to Make Exceptions: Exploring Language Models as Accounts of Human Moral Judgment",
        "Automatic Chain of Thought Prompting in Large Language Models",
        "True Detective: A Deep Abductive Reasoning Benchmark Undoable for GPT-3 and Challenging for GPT-4",
        "Contrastive Chain-of-Thought Prompting",
        "Gemini: A Family of Highly Capable Multimodal Models",
        "Complexity-Based Prompting for Multi-Step Reasoning",
        "Active Prompting with Chain-of-Thought for Large Language Models",
        "MoT: Memory-of-Thought Enables ChatGPT to Self-Improve",
        "Measuring and Narrowing the Compositionality Gap in Language Models",
        "Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data",
        "Tab-CoT: Zero-shot Tabular Chain of Thought",
        "Is a Question Decomposition Unit All We Need?",
        "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
        "Decomposed Prompting: A Modular Approach for Solving Complex Tasks",
        "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models",
        "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "Large Language Model Guided Tree-of-Thought",
        "Cumulative Reasoning with Large Language Models",
        "Graph of thoughts: Solving elaborate problems with large language models",
        "Recursion of Thought: A Divide-and-Conquer Approach to Multi-Context Reasoning with Language Models",
        "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks",
        "Faithful Chain-of-Thought Reasoning",
        "Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding",
        "Exploring Demonstration Ensembling for In-context Learning",
        "$k$NN Prompting: Beyond-Context Learning with Calibration-Free Nearest Neighbor Inference",
        "An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels",
        "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
        "Universal Self-Consistency for Large Language Model Generation",
        "Making Language Models Better Reasoners with Step-Aware Verifier",
        "Language Models (Mostly) Know What They Know",
        "Self-Refine: Iterative Refinement with Self-Feedback",
        "RCOT: Detecting and Rectifying Factual Inconsistency in Reasoning by Reversing Chain-of-Thought",
        "Large Language Models are Better Reasoners with Self-Verification",
        "Deductive Verification of Chain-of-Thought Reasoning",
        "Chain-of-Verification Reduces Hallucination in Large Language Models",
        "Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations",
        "Large Language Models Understand and Can be Enhanced by Emotional Stimuli",
        "Re-Reading Improves Reasoning in Language Models",
        "Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities",
        "Better Zero-Shot Reasoning with Self-Adaptive Prompting",
        "Universal Self-Adaptive Prompting",
        "System 2 Attention (is something you might need too)",
        "Large Language Models as Optimizers",
        "Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves",
    ]

    technique_to_title = {
        "Language Models are Few-Shot Learners": "Few-Shot Learning",
        "A Survey on In-context Learning": "In-context Learning Survey",
        "Exploring Demonstration Ensembling for In-context Learning": "Demonstration Ensembling",
        "Unified Demonstration Retriever for In-Context Learning": "Unified Demo Retriever",
        "Finding Support Examples for In-Context Learning": "Support Examples",
        "Large Language Models Are Human-Level Prompt Engineers": "Human-Level Prompting",
        "Measuring and Narrowing the Compositionality Gap in Language Models": "Compositionality Gap",
        "Automatic Chain of Thought Prompting in Large Language Models": "Automatic CoT",
        "Complexity-Based Prompting for Multi-Step Reasoning": "Complexity-Based Prompting",
        "Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator": "Self-Generated ICL",
        "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models": "Least-to-Most Prompting",
        "Learning To Retrieve Prompts for In-Context Learning": "Prompt Retrieval",
        "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity": "Prompt Order Sensitivity",
        "What Makes Good In-Context Examples for GPT-3?": "Good In-Context Examples",
        "MoT: Memory-of-Thought Enables ChatGPT to Self-Improve": "Memory-of-Thought",
        "kNN Prompting: Beyond-Context Learning with Calibration-Free Nearest Neighbor Inference": "kNN Prompting",
        "Large Language Models are Zero-Shot Reasoners": "Zero-Shot Reasoning",
        "Self-Consistency Improves Chain of Thought Reasoning in Language Models": "Self-Consistency",
        "Large Language Models as Optimizers": "LLMs as Optimizers",
        "Decomposed Prompting: A Modular Approach for Solving Complex Tasks": "Decomposed Prompting",
        "Is a Question Decomposition Unit All We Need?": "Question Decomposition",
        "Deductive Verification of Chain-of-Thought Reasoning": "Deductive Verification",
        "Active Prompting with Chain-of-Thought for Large Language Models": "Active Prompting",
        "Large Language Model Guided Tree-of-Thought": "LLM Guided ToT",
        "Language Models (Mostly) Know What They Know": "LLM Self-Knowledge",
        "Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data": "Automatic Prompt Augmentation",
        "Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations": "Maieutic Prompting",
        "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models": "Plan-and-Solve Prompting",
        "Tree of Thoughts: Deliberate Problem Solving with Large Language Models": "Tree of Thoughts",
        "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks": "Program of Thoughts",
        "Self-Refine: Iterative Refinement with Self-Feedback": "Self-Refine",
        "Cumulative Reasoning with Large Language Models": "Cumulative Reasoning",
        "Faithful Chain-of-Thought Reasoning": "Faithful CoT",
        "Making Language Models Better Reasoners with Step-Aware Verifier": "Step-Aware Verification",
        "Graph of Thoughts: Solving Elaborate Problems with Large Language Models": "Graph of Thoughts",
        "Chain-of-Verification Reduces Hallucination in Large Language Models": "Chain-of-Verification",
        "Better Zero-Shot Reasoning with Self-Adaptive Prompting": "Self-Adaptive Prompting",
        "Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves": "Rephrase and Respond",
    }

    csv_file_path = "path_to_your_csv.csv"
    main.run(csv_file_path, technique_to_title)
