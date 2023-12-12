"""
Test a set of prompts against a dataset and return the results. Currently working for GSM-8k. You must add your openAI API key to the key variable below.
"""

from prompt_systematic_review.role_prompting import evaluate_prompts
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
import json

load_dotenv(dotenv_path="./.env")  # load all entries from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")  # load openai key

with open(
    "data/prompts.json", "r"
) as file:  # load prompts from prompts.json to make prompts more modular.
    prompts = json.load(file)

baseline = prompts["mmlu baseline"]
zero_shot_CoT = prompts["0-shot CoT"]

prompts = [
    baseline,
    baseline + " " + zero_shot_CoT,
]

dataset = "mmlu"  # mmlu or gsm8k
config_name = None  # main if gs8k, None if mmlu
split = "test"
model = "gpt-4-1106-preview"
examples = 1
start = 0  # start index for dataset
log_interval = 25  # log interval for creatings jsons of results by query
max_toks = 4000
rereading = False  # if true, will "reread" the question to the LM at query time
return_json = True
SEED = 42
temp = 0.0

eval = evaluate_prompts(
    prompts,
    dataset,
    config_name,
    split,
    model,
    examples,
    start_index=start,
    log_interval=log_interval,
    max_tokens=max_toks,
    reread=rereading,
    json_mode=return_json,
    seed=SEED,
    temperature=temp,
)

# Getting current date and time in YYYY-MM-DD_HH-MM-SS format
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# File path for the JSON file
file_path = f"data/benchmarking/eval_results_{current_datetime}.json"

# Writing the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(eval, json_file)
