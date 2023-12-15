import pandas as pd
import matplotlib.pyplot as plt

csv_file = "../data/model_citation_counts.csv"
df = pd.read_csv(csv_file)

# sort by alphabetical order
top_20 = df.sort_values(by="model_name", ascending=True)

plt.figure(figsize=(10, 6))
plt.bar(top_20["model_name"], top_20["count"], color="blue")
plt.xlabel("Model Name")
plt.ylabel("Count")
plt.title("Counts of Model Mentions in Dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()
