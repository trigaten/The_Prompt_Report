import os
import json
from prompt_systematic_review.semantic_scholar_source import SemanticScholarSource
from keywords import (
    keywords_list,
)  # Assuming the list of keywords is defined in keywords.py


def create_directory(directory_name):
    """Create a directory if it doesn't already exist."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def save_papers_to_json(papers, file_path):
    """Save a list of Paper objects to a JSON file."""
    papers_dict = [paper.to_dict() for paper in papers]
    with open(file_path, "w") as file:
        json.dump(papers_dict, file, indent=4)


def main():
    sss = SemanticScholarSource()
    flattened_keywords = [keyword for sublist in keywords_list for keyword in sublist]

    directory_name = "papers_output"
    create_directory(directory_name)

    for keyword in flattened_keywords:
        papers = sss.getPapers(300, [keyword])
        file_path = os.path.join(directory_name, f"{keyword.replace(' ', '_')}.json")
        save_papers_to_json(papers, file_path)
        print(f"Saved papers for keyword '{keyword}' to '{file_path}'.")


if __name__ == "__main__":
    main()
