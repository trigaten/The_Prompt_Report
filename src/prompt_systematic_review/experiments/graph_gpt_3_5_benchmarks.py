import matplotlib.pyplot as plt
import os
from prompt_systematic_review.config_data import DataFolderPath


def graph_gpt_3_5():
    # Data for plotting
    data = {
        "math rookie": {"correct": 1230, "total": 2000},
        "knowledgeable AI": {"correct": 1299, "total": 2000},
        "genius...": {"correct": 1174, "total": 2000},
        "idiot...": {"correct": 1218, "total": 2000},
        "careless student": {"correct": 1260, "total": 2000},
        "gardener": {"correct": 1295, "total": 2000},
        "coin that always knows": {"correct": 1277, "total": 2000},
        "mathematician": {"correct": 1262, "total": 2000},
        "farmer": {"correct": 1288, "total": 2000},
        "police officer": {"correct": 1297, "total": 2000},
        "Ivy league math professor": {"correct": 1287, "total": 2000},
        "mentor": {"correct": 1305, "total": 2000},
        "0-Shot CoT": {"correct": 1322, "total": 2000},
        "2-shot CoT": {"correct": 1399, "total": 2000},
        "2-Shot contrastive CoT": {"correct": 1365, "total": 2000},
        "Plan and solve": {"correct": 1364, "total": 2000},
    }

    labels = list(data.keys())
    correct_values = [data[key]["correct"] for key in labels]

    # Creating the bar chart
    plt.figure(figsize=(15, 8))
    bars = plt.bar(labels, correct_values, color="skyblue")

    # Adding labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval, int(yval), ha="center", va="bottom"
        )

    # Customizing the plot
    plt.xlabel("Categories")
    plt.ylabel("Number of Correct Answers")
    plt.title("Correct Answers by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            DataFolderPath,
            "experiments_output" + os.sep + "graph_gpt_3_5_benchmarks_output.pdf",
        ),
        format="pdf",
        bbox_inches="tight",
    )


class Experiment:
    def run():
        graph_gpt_3_5()
