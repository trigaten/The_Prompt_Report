#%% Pip Installs
%pip install tqdm
%pip install pandas
%pip install scipy

# %% RUN 1st
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
# %% RUN 2nd
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

def get_citation_counts(paper_ids):
    """
    Fetches citation counts for a list of paper IDs from Semantic Scholar's API.

    Parameters:
    - paper_ids (list): A list of paper IDs (strings).

    Returns:
    - dict: A dictionary where keys are paper IDs and values are citation counts.
    """
    # Define the URL for the batch API endpoint
    url = 'https://api.semanticscholar.org/graph/v1/paper/batch'
    
    # Define the parameters for the fields you want to retrieve
    params = {'fields': 'citationCount'}
    
    # The payload for the POST request, including the list of paper IDs
    payload = {"ids": paper_ids}
    
    # Make the POST request to the batch API endpoint
    response = requests.post(url, params=params, json=payload)
    
    # Initialize an empty dictionary to store the citation counts
    citation_counts = {}
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Loop through the papers in the response
        for paper in data:
            # Extract the paper ID and citation count
            paper_id = paper.get('paperId')
            citation_count = paper.get('citationCount', 0)
            
            # Add the citation count to the dictionary
            citation_counts[paper_id] = citation_count
    else:
        print(f"Failed to fetch data: {response.status_code}")
    
    return citation_counts



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


# %%Run 3rd
import os
import csv
import json
from tqdm import tqdm
import random

# Path to the CSV file containing the papers' data
csv_file_path = "../../../data/master_papers.csv"

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

# %%RUN 4th
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
    paper_id = query_paper_id(title, api_key)
    references = get_references(paper_id, api_key)

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


# %%Run 5th
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


# %%Run 6th
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


# %%Run - Update File Path
import scipy 
import networkx as nx
import matplotlib.pyplot as plt
import textwrap


# Function to adjust node positions to maintain a min and max distance
def adjust_overlap(pos, nodes_to_adjust, min_dist=1.0, max_dist=2.0, repulsion_factor=1.05, attraction_factor=0.95, vertical_bias=2.0):
    for _ in range(1000):  # Iteration limit
        adjusted = False
        for node1 in nodes_to_adjust:
            for node2 in nodes_to_adjust:
                if node1 == node2:
                    continue  # Skip self comparisons
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                dx, dy = x1 - x2, y1 - y2
                dist = (dx**2 + dy**2) ** 0.5  # Euclidean distance
                
                if dist < min_dist:  # Nodes too close, push apart
                    factor = repulsion_factor
                elif dist > max_dist:  # Nodes too far, pull together
                    factor = -attraction_factor
                else:
                    continue  # Distance is acceptable, no adjustment
                
                if dist == 0:  # Prevent division by zero
                    dx, dy = 1, 0
                    dist = 1
                
                dx, dy = dx / dist * min_dist * factor, dy / dist * min_dist * factor
                dy *= vertical_bias  # Apply vertical adjustment
                
                # Adjust positions
                pos[node1] = (x1 + dx, y1 + dy)
                pos[node2] = (x2 - dx, y2 - dy)
                adjusted = True

        if not adjusted:
            break  # Stop if no adjustments were made in a full pass


# Load the cleaned references
with open("cleaned_complete_paper_references.json","r",
) as json_file:
    paper_references = json.load(json_file)

# Create the graph
G = nx.DiGraph()
for paper_id, references in paper_references.items():
    for ref_id in references:
        G.add_edge(paper_id, ref_id)


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
    "Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves": "Rephrase and Respond"
}
# Find the nodes with at least one incoming edge
nodes_with_incoming_edges = [node for node in G.nodes() if G.in_degree(node) > 0]

top_nodes = sorted(G.nodes(), key=lambda n: G.in_degree(n), reverse=True)[:25]


# Define a function to wrap text into at most two lines
def wrap_text(text, width, max_lines=3):
    wrapped_lines = textwrap.wrap(text, width)
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        wrapped_lines[
            -1
        ] += "..."  # Add ellipsis to the last line to indicate truncation
    return "\n".join(wrapped_lines)


# Assign and label nodes with titles
titles_above_threshold = {}
for paper_id in top_nodes:
    full_title = get_paper_title(paper_id, api_key)  # Fetch full paper title
    if full_title:
        display_title = technique_to_title.get(full_title, full_title)
        wrapped_title = wrap_text(display_title, 10)
        titles_above_threshold[paper_id] = wrapped_title

# Set node sizes proportional to the number of incoming edges (increased size)
node_sizes = [((G.in_degree(node) * 300)+1000) for node in top_nodes]

# Calculate font size based on in-degree (you can adjust the scaling factor)
font_sizes = {node: G.in_degree(node) * 0.06 + 14 for node in top_nodes}

# Draw the graph with adjusted layout parameters
plt.figure(figsize=(50, 30))

pos = nx.spring_layout(G, k=.3, iterations=50, scale=2)  # Initial layout
adjust_overlap(pos, top_nodes, min_dist=1, max_dist=7.5)  # Adjust node positions

# Draw nodes with incoming edges
nx.draw_networkx_nodes(
    G, pos, nodelist=top_nodes, node_size=node_sizes, node_color=(45/255, 137/255, 145/255, 1)
)

# Draw the edges
nx.draw_networkx_edges(G, pos, edge_color="gray", width=0.3)

# Assign and label nodes with titles without y_offset
for node, label in titles_above_threshold.items():
    x, y = pos[node]
    plt.text(x, y, label, fontsize=font_sizes[node], ha="center", va="center", wrap=True)

plt.axis("off")
plt.show()
# # Print the names of all the nodes
# print("Names of all nodes:")
# for node in top_nodes:
#     full_title = get_paper_title(node, api_key)
#     if full_title:
#         print(full_title)

# %%Run 
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
    "Language Models are Few-Shot Learners": "Few-shot Learning (FSL)",
    # "A Survey on In-context Learning": "In-context learning (ICL)",
    "What Makes Good In-Context Examples for {\GPT}-3?": "In-Context Learning (ICL)",
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
with open(
    "cleaned_complete_paper_references.json","r",) as file:
    paper_references = json.load(file)

# Initialize a dictionary for citation counts
citation_counts = {}

# Iterate over the title_to_technique mapping
for title, technique in title_to_technique.items():
    paper_id = query_paper_id(title, api_key)  # Get the paper ID for each title
    # Count how many times each paper_id appears in the reference lists of other papers
    citation_count = sum(paper_id in refs for refs in paper_references.values())
    citation_counts[technique] = citation_count

# Sort the citation counts in descending order
sorted_citations = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)

# Unpack the sorted data for plotting
sorted_techniques, sorted_counts = zip(*sorted_citations)

# Create a vertical bar chart
plt.figure(figsize=(30, 12))
plt.bar(
    sorted_techniques, sorted_counts, color=(45 / 255, 137 / 255, 145 / 255, 1)
)  # RGBA color

plt.yscale("log")

# Rotate the x-axis labels by 45 degrees for better readability
plt.xticks(rotation=45, ha="right")

# Remove the top and right borders
ax = plt.gca()  # Get current axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add labels and title
plt.ylabel("Number of References")
plt.xlabel("Technique")
# plt.title("Internal Citation Counts for Techniques Based on Papers")

plt.tight_layout()  # Adjusts layout to prevent clipping of labels
plt.show()
# %%
