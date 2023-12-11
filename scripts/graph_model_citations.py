import matplotlib.pyplot as plt

# Model names and citation counts, from Semantic Scholar and Google Scholar (for ones which didn't have citation info on Semantic Scholar)
models = [
    "GPT-3",
    "GPT-4",
    "InstructGPT",
    "Codex",
    "BLOOM",
    "BLOOMZ",
    "OPT",
    "LLaMA",
    "Meta AI Codellama",
    "Lambda",
    "PaLM",
    "LLaVA",
    "CODEGEN",
    "GPT-NeoX",
    "SynthIE",
    "FLAN",
    "BERT",
    "RoBERTa",
    "BioBERT",
    "FinBERT",
    "GatorTron",
    "BART",
    "DreamFusion",
    "CLIP",
    "CoCoOp",
    "BLIP-2",
    "Vision Transformer (ViT)",
    "Flamingo",
    "Grounding DINO",
    "YOLOv5m",
    "CLIPSeg",
    "VLP",
    "XMem",
    "SAM",
]

citations = [
    18436,
    1929,
    3731,
    1814,
    922,
    193,
    1472,
    2789,
    133,
    906,
    2669,
    461,
    272,
    376,
    16,
    1452,
    65064,
    16028,
    3657,
    364,
    125,
    6725,
    650,
    9616,
    468,
    868,
    19979,
    1156,
    264,
    35,
    135,
    700,
    108,
    1046,
]


plt.figure(figsize=(10, 6))
plt.bar(models, citations, color="blue")
plt.xlabel("Dataset Name")
plt.ylabel("Number of Citations")
plt.title("Dataset Citations")
plt.xticks(rotation=45, ha="right")

# Display counts on top of each bar
for i, v in enumerate(citations):
    plt.text(i, v + 10, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.show()
