import matplotlib.pyplot as plt

# Dataset names and citation counts, from Semantic Scholar
datasets = [
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
citations = [702, 390, 608, 335, 629, 385, 816, 215, 387, 227, 594]

plt.figure(figsize=(10, 6))
plt.bar(datasets, citations, color="blue")
plt.xlabel("Dataset Name")
plt.ylabel("Number of Citations")
plt.title("Dataset Citations")
plt.xticks(rotation=45, ha="right")

# Display counts on top of each bar
for i, v in enumerate(citations):
    plt.text(i, v + 10, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.show()
