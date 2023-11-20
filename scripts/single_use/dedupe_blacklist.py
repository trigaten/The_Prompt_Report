import pandas as pd
from prompt_systematic_review.utils import process_paper_title

# Load the CSV file
df = pd.read_csv("../../data/blacklist.csv")

df["title"] = df["title"].apply(process_paper_title)

# Get the number of rows before removing duplicates
old_length = len(df)

# Remove duplicates based on 'title' column
df = df.drop_duplicates(subset="title")

# Get the number of rows after removing duplicates
new_length = len(df)

# Print old and new length
print(f"Old length: {old_length}")
print(f"New length: {new_length}")

# Save the DataFrame to the same CSV file
df.to_csv("../../data/blacklist.csv", index=False)
