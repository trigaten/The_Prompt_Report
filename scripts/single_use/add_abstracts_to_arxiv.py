import csv
import pandas as pd
import requests
from xml.etree import ElementTree as ET
import tqdm


# Read the csv data
data = pd.read_csv("../arxiv_papers.csv")

# Create a new column for abstracts
data["abstract"] = ""

# Iterate over each row in the DataFrame
for index, row in tqdm.tqdm(data.iterrows()):
    # Search for the paper on ArXiv API
    # Depending on the format of your URLs, you might need to modify the slice index
    paper_id = row["url"].split("/")[-1].split(".pdf")[0]

    # Create URL for arXiv API call
    url = f"http://export.arxiv.org/api/query?id_list={paper_id}"

    # Fetch details for the given paper_id
    data_response = requests.get(url).content.decode("utf-8", "ignore")

    # Parse XML response using ElementTree
    parser = ET.XMLParser(encoding="utf-8")
    root = ET.fromstring(data_response, parser=parser)

    # Get the abstract (summary) from the XML response
    abstract = root.find(
        "{http://www.w3.org/2005/Atom}entry/{http://www.w3.org/2005/Atom}summary"
    )

    # Add the abstract to the DataFrame
    if abstract is not None:
        data.at[index, "abstract"] = abstract.text

# Write the DataFrame to a new CSV file
data.to_csv("../arxiv_papers_with_abstract.csv", index=False)
