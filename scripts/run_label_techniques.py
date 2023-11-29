import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from prompt_systematic_review.label_techniques import label_techniques

file_path = "./master_papers.csv"

# Read the CSV file into a DataFrame
arxiv_papers_df = pd.read_csv(file_path)

for index, row in arxiv_papers_df.iterrows():
    title = row["title"]
    abstract = row["abstract"]

    print(label_techniques(title, abstract))
