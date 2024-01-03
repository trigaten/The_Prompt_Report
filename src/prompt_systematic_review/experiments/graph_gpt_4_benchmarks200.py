import matplotlib.pyplot as plt
from prompt_systematic_review.config_data import DataFolderPath


def graph_gt_4_benchmarks200():
    # Data
    data = {
        "Role Prompts": {
            "Baseline": 163,
            "Knowledgeable AI": 166,
            "Mentor": 157,
            "Math Rookie": 154,
            "Gardener": 161,
            "Coin that always knows...": 154,
            "Farmer": 157,
            "Genius...": 159,
            "Idiot...": 155,
            "Police officer": 160,
            "High school math teacher": 152,
            "Ivy League math professor": 155,
        },
        "Non-role Prompts": {
            "Rereading": 162,
            "Plan-and-solve": 156,
            "0-shot CoT": 160,
            "2-shot CoT": 160,
            "2-shot Contrastive CoT": 156,
            "10-Shot CoT": 160,
            "10-Shot Contrastive CoT": 158,
        },
    }

    # Preparing data for plotting
    categories = ["Role Prompts", "Non-role Prompts"]
    colors = ["blue", "green"]
    role_prompt_scores = list(data["Role Prompts"].values())
    non_role_prompt_scores = list(data["Non-role Prompts"].values())
    labels = list(data["Role Prompts"].keys()) + list(data["Non-role Prompts"].keys())

    # Sorting the data within each category from highest to lowest score
    sorted_role_prompts = dict(
        sorted(data["Role Prompts"].items(), key=lambda item: item[1], reverse=True)
    )
    sorted_non_role_prompts = dict(
        sorted(data["Non-role Prompts"].items(), key=lambda item: item[1], reverse=True)
    )

    # Preparing sorted data for plotting
    sorted_role_prompt_scores = list(sorted_role_prompts.values())
    sorted_non_role_prompt_scores = list(sorted_non_role_prompts.values())
    sorted_labels = list(sorted_role_prompts.keys()) + list(
        sorted_non_role_prompts.keys()
    )

    # Plotting the sorted data
    plt.figure(figsize=(15, 8))
    bars1 = plt.bar(
        range(len(sorted_role_prompts)),
        sorted_role_prompt_scores,
        color=colors[0],
        label=categories[0],
    )
    bars2 = plt.bar(
        range(len(sorted_role_prompts), len(sorted_labels)),
        sorted_non_role_prompt_scores,
        color=colors[1],
        label=categories[1],
    )

    # Adding the number above each bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                "%d" % int(height),
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.5),
            )

    # Customizing the plot
    plt.xlabel("Prompts")
    plt.ylabel("Scores")
    plt.title("Comparison of Role vs Non-role Prompts (Sorted)")
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=90)
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()


class Experiment:
    def run():
        graph_gt_4_benchmarks200()
