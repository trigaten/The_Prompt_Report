import os
import requests
import time
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import csv
import random
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
import matplotlib


"""
This script processes a CSV file containing research paper information, retrieves reference data from the Semantic Scholar API, 
and visualizes the citation graph and citation counts.

The SemanticScholarAPI class handles interactions with the Semantic Scholar API, including getting paper references and citation counts.
It uses rate limiting and exponential backoff to handle API rate limits and errors.

The ArxivAPI class retrieves paper titles from the arXiv API given an arXiv ID.

The PaperProcessor class processes the CSV file, extracts paper IDs (either directly from Semantic Scholar or by converting arXiv IDs), 
and retrieves the references for each paper using the SemanticScholarAPI. It can also process additional papers given a list of titles.

The GraphVisualizer class visualizes the citation graph, showing the most referenced papers and their connections. 
It applies techniques like adjusting node overlap, wrapping long titles, and scaling node sizes based on the number of incoming citations.
It can also visualize a bar chart of the citation counts for different techniques.

The Main class ties everything together. It loads the API key from a .env file, initializes the other classes, and provides methods to:
1. Process the initial CSV file
2. Process additional papers 
3. Merge the reference data
4. Clean the merged data (remove references to papers not in the dataset)
5. Visualize the citation graph

To use this script:
1. Set your Semantic Scholar API key in a .env file as SEMANTIC_SCHOLAR_API_KEY.
2. Prepare a Master Papers CSV 
3. Update the 'titles' list with any additional papers to include.
4. Update the 'technique_to_title' dictionary to map paper titles to the techniques they represent.
5. Run the script, uncommenting the desired methods in the 'if __name__ == "__main__":' block.

The script will save the processed data to JSON files at each step, and display the visualizations using matplotlib and the graph as a PNG.


May need to uncomment some stuff at the bottom for this to work."""


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
                if response.status_code == 429 or response.status_code == 504:
                    print(f"Received status code {response.status_code}. Retrying...")
                    delay *= 2  # Exponential backoff
                    retries += 1
                else:
                    print(f"HTTPError: {e}")
                    break  # For other HTTP errors, don't retry
            except requests.RequestException as e:
                print(f"Request error: {e}")
                break  # For non-HTTP errors, don't retry
        return None

    def get_references(self, paper_id):
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,authors&limit=1000"
        headers = {"x-api-key": self.api_key}
        response = self.rate_limited_request(url, headers)

        if response and response.status_code == 200:
            data = response.json()
            reference_ids = []

            if "data" in data:
                for ref in data["data"]:
                    if "citedPaper" in ref and "paperId" in ref["citedPaper"]:
                        reference_ids.append(ref["citedPaper"]["paperId"])

            return reference_ids
        return []

    def get_citation_counts(self, paper_ids):
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        params = {"fields": "citationCount"}
        payload = {"ids": paper_ids}
        response = requests.post(url, params=params, json=payload)
        citation_counts = {}

        if response.status_code == 200:
            data = response.json()
            for paper in data:
                paper_id = paper.get("paperId")
                citation_count = paper.get("citationCount", 0)
                citation_counts[paper_id] = citation_count
        else:
            print(f"Failed to fetch data: {response.status_code}")

        return citation_counts

    def query_paper_id(self, title):
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1"
        headers = {"x-api-key": self.api_key}
        response = self.rate_limited_request(url, headers, delay=1, max_retries=4)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and data["data"]:
                return data["data"][0]["paperId"]
        return None

    def get_paper_title(self, paper_id):
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        headers = {"x-api-key": self.api_key}
        response = self.rate_limited_request(url, headers)

        if response and response.status_code == 200:
            data = response.json()
            return data.get("title")
        else:
            print(f"Failed to fetch data for paper ID: {paper_id}")
            return None


class ArxivAPI:
    def get_arxiv_paper_title(self, arxiv_id):
        arxiv_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(arxiv_url)
        if response.status_code != 200:
            return None
        start = response.text.find("<title>") + 7
        end = response.text.find("</title>", start)
        title = response.text[start:end].strip()
        return title


class PaperProcessor:
    def __init__(self, api_key):
        self.semantic_scholar_api = SemanticScholarAPI(api_key)
        self.arxiv_api = ArxivAPI()

    def extract_arxiv_id(self, url):
        base = url.rsplit("/", 1)[-1]
        arxiv_id_with_version = base.split(".pdf")[0]
        arxiv_id = (
            arxiv_id_with_version.split("v")[0]
            if "v" in arxiv_id_with_version
            else arxiv_id_with_version
        )
        return arxiv_id

    def convert_arxiv_id_to_semantic_scholar_id(self, arxiv_id):
        title = self.arxiv_api.get_arxiv_paper_title(arxiv_id)
        if title:
            return self.semantic_scholar_api.query_paper_id(title)
        return None

    def process_papers(self, csv_file_path):
        paper_references = {}
        unmatched_papers = {}

        if os.path.exists(csv_file_path):
            with open(csv_file_path, mode="r", encoding="utf-8") as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=",")
                all_papers = list(csv_reader)
                for row in tqdm(all_papers, desc="Processing Papers"):
                    source = row.get("source", "").strip()
                    if source == "Semantic Scholar":
                        paper_id = row.get("paperId", "").strip()
                    elif source == "arXiv":
                        arxiv_paper_id = self.extract_arxiv_id(
                            row.get("url", "").strip()
                        )
                        paper_id = self.convert_arxiv_id_to_semantic_scholar_id(
                            arxiv_paper_id
                        )
                    else:
                        unmatched_papers[
                            row.get("title", "").strip()
                        ] = "Source not supported"
                        continue

                    if paper_id:
                        references = self.semantic_scholar_api.get_references(paper_id)
                        if references is not None:
                            paper_references[paper_id] = references
                        else:
                            unmatched_papers[
                                row["title"]
                            ] = "No references found or error occurred"
                    else:
                        print(f"Paper Id Could not be found for: {row}")
        else:
            print(f"CSV file does not exist: {csv_file_path}")

        return paper_references, unmatched_papers

    def process_additional_papers(self, titles):
        paper_references = {}
        unmatched_titles = {}

        for title in tqdm(titles):
            paper_id = self.semantic_scholar_api.query_paper_id(title)
            references = self.semantic_scholar_api.get_references(paper_id)

            if paper_id:
                paper_references[paper_id] = references
            else:
                unmatched_titles[title] = "No matching paper ID found"

        return paper_references, unmatched_titles


class GraphVisualizer:
    def __init__(self, api_key):
        self.semantic_scholar_api = SemanticScholarAPI(api_key)

    def adjust_overlap(
        self,
        pos,
        nodes_to_adjust,
        min_dist=1.0,
        max_dist=2.0,
        repulsion_factor=1.05,
        attraction_factor=0.95,
        vertical_bias=2.0,
    ):
        for _ in range(1000):
            adjusted = False
            for node1 in nodes_to_adjust:
                for node2 in nodes_to_adjust:
                    if node1 == node2:
                        continue
                    x1, y1 = pos[node1]
                    x2, y2 = pos[node2]
                    dx, dy = x1 - x2, y1 - y2
                    dist = (dx**2 + dy**2) ** 0.5

                    if dist < min_dist:
                        factor = repulsion_factor
                    elif dist > max_dist:
                        factor = -attraction_factor
                    else:
                        continue

                    if dist == 0:
                        dx, dy = 1, 0
                        dist = 1

                    dx, dy = (
                        dx / dist * min_dist * factor,
                        dy / dist * min_dist * factor,
                    )
                    dy *= vertical_bias

                    pos[node1] = (x1 + dx, y1 + dy)
                    pos[node2] = (x2 - dx, y2 - dy)
                    adjusted = True

            if not adjusted:
                break

    def wrap_text(self, text, width, max_lines=3):
        wrapped_lines = textwrap.wrap(text, width)
        if len(wrapped_lines) > max_lines:
            wrapped_lines = wrapped_lines[:max_lines]
            wrapped_lines[-1] += "..."
        return "\n".join(wrapped_lines)

    def visualize_graph(self, paper_references, technique_to_title):
        G = nx.DiGraph()
        for paper_id, references in paper_references.items():
            for ref_id in references:
                G.add_edge(paper_id, ref_id)

        nodes_with_incoming_edges = [
            node for node in G.nodes() if G.in_degree(node) > 0
        ]

        top_nodes = sorted(G.nodes(), key=lambda n: G.in_degree(n), reverse=True)[:25]

        titles_above_threshold = {}
        for paper_id in top_nodes:
            full_title = self.semantic_scholar_api.get_paper_title(paper_id)
            if full_title:
                display_title = technique_to_title.get(full_title, full_title)
                wrapped_title = self.wrap_text(display_title, 10)
                titles_above_threshold[paper_id] = wrapped_title

        node_sizes = [((G.in_degree(node) * 80) + 50) for node in top_nodes]

        font_sizes = {node: G.in_degree(node) * 0.001 + 12 for node in top_nodes}

        plt.figure(figsize=(10, 6))

        pos = nx.spring_layout(G, k=0.3, iterations=50, scale=2)
        self.adjust_overlap(pos, top_nodes, min_dist=0.5, max_dist=5)

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=top_nodes,
            node_size=node_sizes,
            node_color=(45 / 255, 137 / 255, 145 / 255, 1),
        )

        nx.draw_networkx_edges(G, pos, edge_color="gray", width=0.3)

        for node, label in titles_above_threshold.items():
            x, y = pos[node]
            plt.text(
                x,
                y,
                label,
                fontsize=font_sizes[node],
                ha="center",
                va="center",
                wrap=True,
            )

        plt.axis("off")

        # plt.show()
        plt.savefig("network_graph.png", format="png", dpi=300)

    def visualize_citation_counts(self, paper_references, title_to_technique):
        citation_counts = {}
        for title, technique in title_to_technique.items():
            paper_id = self.semantic_scholar_api.query_paper_id(title)
            citation_count = sum(paper_id in refs for refs in paper_references.values())
            citation_counts[technique] = citation_count

        sorted_citations = sorted(
            citation_counts.items(), key=lambda x: x[1], reverse=True
        )
        sorted_techniques, sorted_counts = zip(*sorted_citations)

        plt.figure(figsize=(15, 6))
        plt.bar(
            sorted_techniques, sorted_counts, color=(45 / 255, 137 / 255, 145 / 255, 1)
        )
        plt.yscale("log")
        plt.xticks(rotation=45, ha="right")
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.ylabel("Citation Counts")
        plt.xlabel("Prompting Techniques")
        plt.title("Citation Counts of Prompting Techniques")
        plt.tight_layout()
        plt.show()


class Main:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.paper_processor = PaperProcessor(self.api_key)
        self.graph_visualizer = GraphVisualizer(self.api_key)

    def process_papers(self, csv_file_path):
        paper_references, unmatched_papers = self.paper_processor.process_papers(
            csv_file_path
        )
        with open("revised_paper_references.json", "w") as json_file:
            json.dump(paper_references, json_file, indent=4)
        print("Data saved to revised_paper_references.json")

    def process_additional_papers(self, titles):
        (
            paper_references,
            unmatched_titles,
        ) = self.paper_processor.process_additional_papers(titles)
        with open("paper_references_additional.json", "w") as json_file:
            json.dump(paper_references, json_file, indent=4)
        print("Data saved to paper_references_additional.json")
        with open("unmatched_titles_additional.json", "w") as json_file:
            json.dump(unmatched_titles, json_file, indent=4)
        print("Data saved to unmatched_titles_additional.json")

    def merge_paper_references(self):
        with open("revised_paper_references.json", "r") as file:
            paper_references = json.load(file)
        with open("paper_references_additional.json", "r") as file:
            paper_references_additional = json.load(file)
        paper_references.update(paper_references_additional)
        with open("complete_paper_references.json", "w") as file:
            json.dump(paper_references, file, indent=4)
        print("Merged data saved to complete_paper_references.json")

    def clean_paper_references(self):
        with open(
            "complete_paper_references.json",
            "r",
        ) as file:
            merged_paper_references = json.load(file)
        for paper_id, references in merged_paper_references.items():
            merged_paper_references[paper_id] = [
                ref for ref in references if ref in merged_paper_references
            ]
        with open("cleaned_complete_paper_references.json", "w") as file:
            json.dump(merged_paper_references, file, indent=4)
        print("Cleaned data saved to cleaned_merged_paper_references.json")

    def visualize_graph(self, technique_to_title):
        with open("cleaned_complete_paper_references.json", "r") as json_file:
            paper_references = json.load(json_file)
        self.graph_visualizer.visualize_graph(paper_references, technique_to_title)

    def visualize_chart(self, technique_to_title):
        with open("cleaned_complete_paper_references.json", "r") as json_file:
            paper_references = json.load(json_file)
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
    main.visualize_chart(technique_to_title)
