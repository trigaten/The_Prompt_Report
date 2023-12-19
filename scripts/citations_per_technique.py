import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, LogFormatter

# Data
data = {
    "Technique": [
    "Zero-Shot Prompt", "Role Prompting", "Style Prompting", "In-context learning (ICL)", "Few-shot learning (FSL)", "K-Nearest Neighbor (KNN)",
     "fiLter-thEN-Search (LENS)", "Unified Demonstration Retriever (UDR)", "Example Ordering", "Self-Generated In-context Learning (SG-ICL)", "Example Number", 
     "Example Label Quality", "Input Distribution", "Input-Label Pairing Format", "Zero-shot-CoT", "Step-Back Prompting", "Thread-of-Thought (ThoT)", 
     "Moral Chain-of-Thought", "Few-Shot CoT", "Contrastive Chain of Thought", "Uncertainty-Routed CoT", "Complexity-based Prompting", "Active-Prompt", 
     "Memory-of-Thought", "Self-Ask", "Automatic Chain-of-Thought Prompting (Auto-CoT)", "Automate-CoT", "Tab-CoT", "Decomposition", "Least-to-most Prompting", 
     "Decomposed Prompting (DECOMP)", "Plan-and-Solve Prompting", "Tree-of-Thought (ToT)", "Cumulative Reasoning", "Graph-of-Thoughts", "Recursion of Thought", 
     "Program-of-Thoughts", "Faithful Chain-of-Thought", "Skeleton-of-Thought", "Demonstration Ensembling (DENSE)", "Max Mutual Information Method", "Self-Consistency", 
     "Universal Self-Consistency", "DiVeRSe", "Self-Evaluation", "Self-Refine", "Reversing Chain-of-Thought (RCoT)", "Self Verification", "Deductive Verification",
      "Chain-of-Verification (COVE)", "Maieutic Prompting", "Prompt Mining", "EmotionPrompt", "Re-reading (RE2)", "Think Twice (SIMTOM)", 
      "Consistency-based Self-adaptive Prompting (COSP)", "Universal Self-Adaptive Prompting (USP)", "System 2 Attention", "OPRO", "Rephrase and Respond (RAR)"
    ],
    "Parent Category": [
    "Zero-Shot In-Context Learning", "Zero-Shot In-Context Learning", "Zero-Shot In-Context Learning", "Few-Shot In-Context Learning", 
    "Few-Shot In-Context Learning", "Example Selection", "Example Selection", "Example Selection", "Example Ordering", "Example Generation",
     "Additional Factors", "Additional Factors", "Additional Factors", "Additional Factors", "Thought Generation", "Thought Generation", 
     "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", 
     "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", 
     "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", 
     "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", "Thought Generation", "Ensembling", "Ensembling",
      "Ensembling", "Ensembling", "Ensembling", "Self-Criticism", "Self-Criticism", "Self-Criticism", "Self-Criticism", "Self-Criticism", 
      "Self-Criticism", "Self-Criticism", "Other", "Other", "Other", "Other", "Other", "Other", "Other", "Other", "Other"
    ],
    "Total Citations": [
        1, 46, 5, 17775, 253, 541, 13, 13, 1232, 16, 18316, 501, 77, 0, 262, 0, 0, 34, 202, 0, 0, 112, 45, 2, 38, 200, 24, 6, 21, 385, 110, 15, 
        291, 19, 53, 3, 144, 17, 7, 14, 54, 358, 0, 27, 81, 185, 0, 33, 19, 21, 17, 876, 0, 4, 0, 11, 3, 0, 50, 0
    ]}
# Ensure all lists are of the same length
print(len(data["Technique"]))
print(len(data["Parent Category"]))
print(len(data["Total Citations"]))

assert len(data["Technique"]) == len(data["Parent Category"]) == len(data["Total Citations"]), "All lists must be of the same length"


# Create DataFrame
df = pd.DataFrame(data)

# Sorting the DataFrame by 'Total Citations' for better visualization
df_sorted = df.sort_values(by='Total Citations', ascending=False)

# Assigning colors to each 'Parent Category'
parent_categories = df_sorted['Parent Category']. unique()
colors = plt.cm.get_cmap('tab20', len(parent_categories))
category_colors = {category: colors(i) for i, category in enumerate(parent_categories)}

# Plotting with vertical bars and rotated Y-axis labels
plt.figure(figsize=(12, 8))  # Increase the figure size as needed
plt.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.15)  # Adjust the margins as needed
bars = plt.bar(df_sorted['Technique'], df_sorted['Total Citations'], color=[category_colors[cat] for cat in df_sorted['Parent Category']], label=df_sorted['Parent Category'])

# Apply logarithmic scale with base 10 to the Y-axis
plt.yscale('log', base=10)
plt.gca().yaxis.set_major_locator(LogLocator(base=10))
plt.gca().yaxis.set_major_formatter(LogFormatter(base=10))

# Add legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Set X and Y axis labels
plt.xlabel('Technique')
plt.ylabel('Log Scale of Total Citations (Base 10)')

plt.title('Citations by Prompting Technique')

# Rotate Y-axis labels for better readability
plt.xticks(rotation=90)

# Adjust layout and margins
plt.tight_layout()

plt.show()