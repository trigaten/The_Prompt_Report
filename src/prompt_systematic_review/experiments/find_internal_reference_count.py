# %%
import os
import requests
import time
import json
from tqdm import tqdm
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


# %%
def rate_limited_request(url, headers, max_retries=3, delay=0.1):
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


# Function to get references from semantic scholar Id
def get_references(paper_id, api_key):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,authors&limit=1000"
    headers = {"x-api-key": api_key}
    response = rate_limited_request(url, headers)

    if response and response.status_code == 200:
        data = response.json()
        reference_ids = []

        if "data" in data:
            for ref in data["data"]:
                if "citedPaper" in ref and "paperId" in ref["citedPaper"]:
                    reference_ids.append(ref["citedPaper"]["paperId"])

        return reference_ids
    return []


# Function to query paper title from Arxiv Id
def get_arxiv_paper_title(arxiv_id):
    arxiv_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(arxiv_url)
    if response.status_code != 200:
        return None
    start = response.text.find("<title>") + 7
    end = response.text.find("</title>", start)
    title = response.text[start:end].strip()
    return title


# Function to query Semantic Scholar for a paper ID using title
def query_paper_id(title, api_key):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1"
    headers = {"x-api-key": api_key}
    response = rate_limited_request(url, headers, delay=1, max_retries=4)
    if response.status_code == 200:
        data = response.json()
        if "data" in data and data["data"]:
            return data["data"][0]["paperId"]
    return None


# Function to convert arXiv ID into Semantic Scholar ID
def convert_arxiv_id_to_semantic_scholar_id(arxiv_id, api_key):
    title = get_arxiv_paper_title(arxiv_id)
    if title:
        return query_paper_id(title, api_key)
    return None


# Function to get the title for a given paper ID
def get_paper_title(paper_id, api_key):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    headers = {"x-api-key": api_key}
    response = rate_limited_request(url, headers)

    if response and response.status_code == 200:
        data = response.json()
        return data.get("title")
    else:
        print(f"Failed to fetch data for paper ID: {paper_id}")
        return None


# %%
import os
import csv
import json
from tqdm import tqdm
import random


# Path to the CSV file containing the papers' data
csv_file_path = "master_papers.csv"

# Dictionary to hold the references
paper_references = {}
unmatched_papers = {}


# Function to extract Arxiv ID from the URL
def extract_arxiv_id(url):
    # Extract the part of the URL after 'pdf/' and before '.pdf'
    base = url.rsplit("/", 1)[-1]
    arxiv_id_with_version = base.split(".pdf")[0]
    # Remove the version part if it exists (e.g., 'v1')
    arxiv_id = (
        arxiv_id_with_version.split("v")[0]
        if "v" in arxiv_id_with_version
        else arxiv_id_with_version
    )
    return arxiv_id


if os.path.exists(csv_file_path):
    with open(csv_file_path, mode="r", encoding="utf-8") as csvfile:
        # Let csv.DictReader read the headers directly from the file
        csv_reader = csv.DictReader(csvfile, delimiter=",")

        # Convert the csv_reader to a list to allow for random sampling
        all_papers = list(csv_reader)
        # random_papers = random.sample(all_papers, min(20, len(all_papers)))  # Safely sample 20 papers or the total count if less
        for row in tqdm(all_papers, desc="Processing Papers"):
            source = row.get(
                "source", ""
            ).strip()  # Use .get() to avoid KeyError and strip() to remove any extra whitespace
            if source == "Semantic Scholar":
                paper_id = row.get("paperId", "").strip()  # Use .get() here as well
            elif source == "arXiv":
                arxiv_paper_id = extract_arxiv_id(row.get("url", "").strip())
                paper_id = convert_arxiv_id_to_semantic_scholar_id(
                    arxiv_paper_id, api_key=api_key
                )
            else:
                unmatched_papers[row.get("title", "").strip()] = "Source not supported"
                continue
            # print(paper_id)
            if paper_id:
                references = get_references(paper_id, api_key=api_key)
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

# Save the results to JSON files
with open("revised_paper_references.json", "w") as json_file:
    json.dump(paper_references, json_file, indent=4)
print("Data saved to revised_paper_references.json")

# %%
# Second main to add important papers not in our original dataset


paper_references = {}
unmatched_titles = {}

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

# Processing the papers with a progress bar
for title in tqdm(titles):
    paper_id, references = get_references(title, api_key)
    if paper_id:
        paper_references[paper_id] = references
    else:
        unmatched_titles[title] = "No matching paper ID found"

# Save the results to JSON files
with open("paper_references_additional.json", "w") as json_file:
    json.dump(paper_references, json_file, indent=4)
print("Data saved to paper_references_additional.json")

with open("unmatched_titles_additional.json", "w") as json_file:
    json.dump(unmatched_titles, json_file, indent=4)
print("Data saved to unmatched_titles_additional.json")


# %%
# Merge the two files

# Load the existing data from both JSON files
with open("revised_paper_references.json", "r") as file:
    paper_references = json.load(file)

with open("paper_references_additional.json", "r") as file:
    paper_references_additional = json.load(file)


# Merge the two dictionaries
# If there are duplicate keys, the values from paper_references_additional will be used
paper_references.update(paper_references_additional)

# Save the merged data back to a JSON file
with open("complete_paper_references.json", "w") as file:
    json.dump(paper_references, file, indent=4)

print("Merged data saved to complete_paper_references.json")


# %%
# Only keep refernces which refer to papers in our combined dataset

# Load the merged paper references
with open("complete_paper_references.json", "r") as file:
    merged_paper_references = json.load(file)

# Filter the references so that only those that are keys in the dictionary are kept
for paper_id, references in merged_paper_references.items():
    merged_paper_references[paper_id] = [
        ref for ref in references if ref in merged_paper_references
    ]

# Save the cleaned data back to a JSON file
with open("cleaned_complete_paper_references.json", "w") as file:
    json.dump(merged_paper_references, file, indent=4)

print("Cleaned data saved to cleaned_merged_paper_references.json")


# %%
# Histogram of the top 30 most cited papers by internal citation count

import json
import matplotlib.pyplot as plt
import textwrap
import matplotlib

# Set font properties globally
matplotlib.rcParams.update(
    {
        "font.family": "Arial",  # You can replace 'Arial' with 'Helvetica' or another modern font
        "font.size": 8,  # You can adjust this value as needed
    }
)

# Load the data
with open("cleaned_complete_paper_references.json", "r") as file:
    data = json.load(file)

# Count the number of references for each key
reference_counts = {key: len(references) for key, references in data.items()}

# Sort by the number of references and select the top 50
top_25 = sorted(reference_counts.items(), key=lambda x: x[1], reverse=True)[:30]

# Fetch the titles for the top 50 paper IDs
top_25_titles = {paper_id: get_paper_title(paper_id, api_key) for paper_id, _ in top_25}

# Replace paper IDs with their titles in the top_50 list
top_25_with_titles = [(top_25_titles[paper_id], count) for paper_id, count in top_25]

# Unpack the data for plotting
titles, counts = zip(*top_25_with_titles)

# Define the RGBA color
rgba_color = (45 / 255, 137 / 255, 145 / 255, 1)  # Converted from rgba(45, 137, 145, 1)

# Create a vertical bar chart
plt.figure(figsize=(12, 8))
plt.bar(titles, counts, color=rgba_color)

# Rotate the x-axis labels by 45 degrees for better readability
plt.xticks(rotation=45, ha="right")  # ha='right' aligns the labels at the right angle

# Add labels and title
plt.ylabel("Number of References")
plt.xlabel("Paper Title")
plt.title("Top 30 Papers by Number of References")

# plt.tight_layout()  # Adjusts layout to prevent clipping of labels
plt.show()


# %%
# Generate a graph of all the papers with more than 10 internal references

import networkx as nx
import matplotlib.pyplot as plt
import textwrap

# Load the cleaned references
with open("cleaned_complete_paper_references.json", "r") as json_file:
    paper_references = json.load(json_file)

# Create the graph
G = nx.DiGraph()
for paper_id, references in paper_references.items():
    for ref_id in references:
        G.add_edge(paper_id, ref_id)

# Remove isolated nodes and nodes with less than 10 incoming edges
G.remove_nodes_from(list(nx.isolates(G)))
nodes_to_remove = [node for node in G.nodes() if G.in_degree(node) < 10]
G.remove_nodes_from(nodes_to_remove)

# Find the top 20 nodes with the most incoming edges
top_nodes = sorted(G.nodes(), key=lambda n: G.in_degree(n), reverse=True)[:20]

titles_above_threshold = {}
for paper_id in top_nodes:
    title = get_paper_title(paper_id, api_key)
    if title:
        if len(title) > 50:
            title = textwrap.shorten(
                title, width=50, placeholder="..."
            )  # Shorten if longer than 60s
        titles_above_threshold[paper_id] = title

# Cap the maximum node size to prevent too large nodes
# max_size = 100000  # Maximum size for a node
node_sizes = [G.in_degree(node) * 2000 for node in G.nodes()]

# Draw the graph with adjusted layout parameters
plt.figure(figsize=(60, 35))  # Increased figure size for more space
pos = nx.kamada_kawai_layout(G, dist=None, scale=1)  # Adjust 'scale' as needed
nx.draw(
    G,
    pos,
    with_labels=False,
    node_size=node_sizes,
    node_color="skyblue",
    edge_color="gray",
    width=0.5,
)

# Assign and label top nodes with titles
for node, label in titles_above_threshold.items():
    x, y = pos[node]
    wrapped_label = label.split("\n")
    for i, line in enumerate(wrapped_label):
        plt.text(x, y, line, fontsize=18, ha="center", va="center")

plt.title("Graph of Paper References")
plt.show()


# %%
import networkx as nx
import matplotlib.pyplot as plt
import textwrap


def adjust_overlap(pos, nodes_to_adjust, min_dist=0.1, repulsion_factor=1.05):
    for _ in range(1000):  # Increase the number of iterations for a denser graph
        adjusted = False
        for i, node1 in enumerate(nodes_to_adjust):
            for node2 in nodes_to_adjust[i + 1 :]:
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                dx, dy = x1 - x2, y1 - y2
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < min_dist:  # If nodes are too close, push them apart
                    # Apply a repulsion factor to move nodes further apart
                    if dist == 0:  # To avoid division by zero
                        dx, dy = 1, 0
                        dist = 1
                    dx, dy = (
                        dx / dist * min_dist * repulsion_factor,
                        dy / dist * min_dist * repulsion_factor,
                    )
                    pos[node1] = (x1 + dx, y1 + dy)
                    pos[node2] = (x2 - dx, y2 - dy)
                    adjusted = True
        if not adjusted:  # Break the loop if no adjustments were made
            break


# Load the cleaned references
with open("cleaned_complete_paper_references.json", "r") as json_file:
    paper_references = json.load(json_file)

# Create the graph
G = nx.DiGraph()
for paper_id, references in paper_references.items():
    for ref_id in references:
        G.add_edge(paper_id, ref_id)

# Remove isolated nodes and nodes with less than 10 incoming edges
G.remove_nodes_from(list(nx.isolates(G)))
nodes_to_remove = [node for node in G.nodes() if G.in_degree(node) < 8]
G.remove_nodes_from(nodes_to_remove)

# Find the top 20 nodes with the most incoming edges
top_nodes = sorted(G.nodes(), key=lambda n: G.in_degree(n), reverse=True)[:25]


# Define a function to wrap text into at most two lines
def wrap_text(text, width, max_lines=2):
    wrapped_lines = textwrap.wrap(text, width)
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        wrapped_lines[
            -1
        ] += "..."  # Add ellipsis to the last line to indicate truncation
    return "\n".join(wrapped_lines)


titles_above_threshold = {}
for paper_id in top_nodes:
    title = get_paper_title(
        paper_id, api_key
    )  # Function to fetch paper title using paper_id
    if title:
        wrapped_title = wrap_text(title, 40)  # Wrap the title into at most two lines
        titles_above_threshold[paper_id] = wrapped_title


# Cap the maximum node size to prevent too large nodes
node_sizes = [G.in_degree(node) * 2000 for node in G.nodes()]

# Draw the graph with adjusted layout parameters
plt.figure(figsize=(60, 35))
pos = nx.kamada_kawai_layout(G, dist=None, scale=1)

adjust_overlap(pos, top_nodes, min_dist=0.2, repulsion_factor=1.05)

# Draw all nodes first
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue")

# Draw the edges
nx.draw_networkx_edges(G, pos, edge_color="gray", width=0.5)

# Assign and label top nodes with titles
for node, label in titles_above_threshold.items():
    x, y = pos[node]
    num_lines = label.count("\n") + 1
    y_offset = 0.01 * 2  # Adjust this factor as needed to position the text correctly
    plt.text(x, y + y_offset, label, fontsize=20, ha="center", va="center")

plt.title("Graph of Paper References")
plt.show()

# %%
# Generate a graph of all the papers with more than 10 internal references
import json
import matplotlib.pyplot as plt
import matplotlib

# Set font properties globally
matplotlib.rcParams.update({"font.family": "Arial", "font.size": 10})

# Mapping of paper titles to their techniques
title_to_technique = {
    "A Practical Survey on Zero-Shot Prompt Design for In-Context Learning": "Zero-Shot Prompt",
    "Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm": "Role Prompting",
    "Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints": "Style Prompting",
    "Language Models are Few-Shot Learners": "In-context learning (ICL)",
    "A Survey on In-context Learning": "Few-shot learning (FSL)",
    "What Makes Good In-Context Examples for {GPT}-3?": "K-Nearest Neighbor (KNN)",
    "Finding Support Examples for In-Context Learning": "fiLter-thEN-Search (LENS)",
    "Unified Demonstration Retriever for In-Context Learning": "Unified Demonstration Retriever (UDR)",
    "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity": "Example Ordering",
    "Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator": "Example Generation",
    "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?": "Example Label Quality",
    "Selective Annotation Makes Language Models Better Few-Shot Learners": "Input Distribution",
    "Chain-of-thought prompting elicits reasoning in large language models": "Chain-of-Thought",
    "Large Language Models are Zero-Shot Reasoners": "Zero-shot-CoT",
    "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models": "Step-Back Prompting",
    "Thread of Thought Unraveling Chaotic Contexts": "Thread-of-Thought (ThoT)",
    "When to Make Exceptions: Exploring Language Models as Accounts of Human Moral Judgment": "Moral Chain-of-Thought",
    "Automatic Chain of Thought Prompting in Large Language Models": "Few-Shot CoT",
    "True Detective: A Deep Abductive Reasoning Benchmark Undoable for GPT-3 and Challenging for GPT-4": "Del_2023",
    "Contrastive Chain-of-Thought Prompting": "Contrastive Chain of Thought",
    "Gemini: A Family of Highly Capable Multimodal Models": "Uncertainty-Routed CoT",
    "Complexity-Based Prompting for Multi-Step Reasoning": "Complexity-based Prompting",
    "Active Prompting with Chain-of-Thought for Large Language Models": "Active-Prompt",
    "MoT: Memory-of-Thought Enables ChatGPT to Self-Improve": "Memory-of-Thought",
    "Measuring and Narrowing the Compositionality Gap in Language Models": "Self-Ask",
    "Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data": "Automate-CoT",
    "Tab-CoT: Zero-shot Tabular Chain of Thought": "Tab-CoT",
    "Is a Question Decomposition Unit All We Need?": "Decomposition",
    "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models": "Least-to-most Prompting",
    "Decomposed Prompting: A Modular Approach for Solving Complex Tasks": "Decomposed Prompting (DECOMP)",
    "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models": "Plan-and-Solve Prompting",
    "Tree of Thoughts: Deliberate Problem Solving with Large Language Models": "Tree-of-Thought (ToT)",
    "Cumulative Reasoning with Large Language Models": "Cumulative Reasoning",
    "Graph of thoughts: Solving elaborate problems with large language models": "Graph-of-Thoughts",
    "Recursion of Thought: A Divide-and-Conquer Approach to Multi-Context Reasoning with Language Models": "Recursion of Thought",
    "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks": "Program-of-Thoughts",
    "Faithful Chain-of-Thought Reasoning": "Faithful Chain-of-Thought",
    "Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding": "Skeleton-of-Thought",
    "Exploring Demonstration Ensembling for In-context Learning": "Demonstration Ensembling (DENSE)",
    "$k$NN Prompting: Beyond-Context Learning with Calibration-Free Nearest Neighbor Inference": "kNN Prompting",
    "An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels": "Max Mutual Information Method",
    "Self-Consistency Improves Chain of Thought Reasoning in Language Models": "Self-Consistency",
    "Universal Self-Consistency for Large Language Model Generation": "Universal Self-Consistency",
    "Making Language Models Better Reasoners with Step-Aware Verifier": "DiVeRSe",
    "Language Models (Mostly) Know What They Know": "Self-Evaluation",
    "Self-Refine: Iterative Refinement with Self-Feedback": "Self-Refine",
    "RCOT: Detecting and Rectifying Factual Inconsistency in Reasoning by Reversing Chain-of-Thought": "Reversing Chain-of-Thought (RCoT)",
    "Large Language Models are Better Reasoners with Self-Verification": "Self Verification",
    "Deductive Verification of Chain-of-Thought Reasoning": "Deductive Verification",
    "Chain-of-Verification Reduces Hallucination in Large Language Models": "Chain-of-Verification (COVE)",
    "Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations": "Maieutic Prompting",
    "How can we know what language models know?": "Prompt Mining",
    "Large Language Models Understand and Can be Enhanced by Emotional Stimuli": "EmotionPrompt",
    "Re-Reading Improves Reasoning in Language Models": "Re-reading (RE2)",
    "Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities": "Think Twice (SIMTOM)",
    "Better Zero-Shot Reasoning with Self-Adaptive Prompting": "Consistency-based Self-adaptive Prompting (COSP)",
    "Universal Self-Adaptive Prompting": "Universal Self-Adaptive Prompting (USP)",
    "System 2 Attention (is something you might need too)": "System 2 Attention",
    "Large Language Models as Optimizers": "OPRO",
    "Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves": "Rephrase and Respond (RAR)",
}

# Load the existing dictionary of paper references
with open("cleaned_complete_paper_references.json", "r") as file:
    paper_references = json.load(file)

# Query each title and get citation counts
citation_counts = {}
for title, technique in title_to_technique.items():
    paper_id = query_paper_id(title, api_key)
    if paper_id and paper_id in paper_references:
        citation_count = len(paper_references[paper_id])
        citation_counts[technique] = citation_count
        print({technique}, {citation_count})

# Sort the citation counts in descending order
sorted_citations = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)

# Unpack the sorted data for plotting
sorted_techniques, sorted_counts = zip(*sorted_citations)

# Create a vertical bar chart
plt.figure(figsize=(30, 12))
plt.bar(
    sorted_techniques, sorted_counts, color=(45 / 255, 137 / 255, 145 / 255, 1)
)  # RGBA color

# Rotate the x-axis labels by 45 degrees for better readability
plt.xticks(rotation=90, ha="right")  # ha='right' aligns the labels at the right angle

# Add labels and title
plt.ylabel("Number of References")
plt.xlabel("Technique")
plt.title("Citation Counts for Techniques Based on Papers")

plt.tight_layout()  # Adjusts layout to prevent clipping of labels
plt.show()
