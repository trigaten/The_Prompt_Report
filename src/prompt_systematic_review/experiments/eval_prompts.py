"""
Test a set of prompts against a dataset and return the results. Currently working for GSM-8k. You must add your openAI API key to the key variable below.
"""

from prompt_systematic_review.benchmarking import evaluate_prompts
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
import json
from prompt_systematic_review.benchmarking import Prompt
from prompt_systematic_review.config_data import DataFolderPath, DotenvPath

load_dotenv(dotenv_path=DotenvPath)  # load all entries from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")  # load openai key

with open(
    "data/prompts.json", "r"
) as file:  # load prompts from prompts.json to make prompts more modular.
    prompts = json.load(file)

baseline_prompts = [
    prompts["baseline1"],
    prompts["baseline2"],
    prompts["baseline3"],
]  # creating list of baseline prompts ex. "Solve this problem..."
zero_shot_cot_prompts = [
    prompts["let's think..."],
    prompts["plan-and-solve"],
    prompts["thread-of-thought"],
]

# creating prompt objects
zero_shot_baseline1_format1 = Prompt(
    "0-Shot Vanilla 1 Format 1", baseline_prompts[0], 1
)
zero_shot_baseline1_format2 = Prompt(
    "0-shot Vanilla 1 Format 2", baseline_prompts[0], 2
)

zero_shot_baseline2_format1 = Prompt(
    "0-Shot Vanilla 2 Format 1", baseline_prompts[1], 1
)
zero_shot_baseline2_format2 = Prompt(
    "0-shot Vanilla 2 Format 2", baseline_prompts[1], 2
)

zero_shot_baseline3_format1 = Prompt(
    "0-shot Vanilla 3 Format 1", baseline_prompts[2], 1
)
zero_shot_baseline3_format2 = Prompt(
    "0-shot Vanilla 3 Format 2", baseline_prompts[2], 2
)


zero_shot_CoT1_format1 = Prompt(
    "0-Shot CoT 1 Format 1", zero_shot_cot_prompts[0], 1, CoT=True
)
zero_shot_CoT1_format2 = Prompt(
    "0-Shot CoT 1 Format 2", zero_shot_cot_prompts[0], 2, CoT=True
)

zero_shot_CoT2_format1 = Prompt(
    "0-Shot CoT 2 Format 1", zero_shot_cot_prompts[1], 1, CoT=True
)
zero_shot_CoT2_format2 = Prompt(
    "0-Shot CoT 2 Format 2", zero_shot_cot_prompts[1], 2, CoT=True
)

zero_shot_CoT3_format1 = Prompt(
    "0-Shot CoT 3 Format 1", zero_shot_cot_prompts[2], 1, CoT=True
)
zero_shot_CoT3_format2 = Prompt(
    "0-Shot CoT 3 Format 2", zero_shot_cot_prompts[2], 2, CoT=True
)

few_shot_baseline1_format1 = Prompt(
    "Few-Shot Vanilla 1 Format 1", baseline_prompts[0], 1, shots=True
)
few_shot_baseline1_format2 = Prompt(
    "Few-Shot Vanilla 1 Format 2", baseline_prompts[0], 2, shots=True
)

few_shot_baseline2_format1 = Prompt(
    "Few-Shot Vanilla 2 Format 1", baseline_prompts[1], 1, shots=True
)
few_shot_baseline2_format2 = Prompt(
    "Few-Shot Vanilla 2 Format 2", baseline_prompts[1], 2, shots=True
)

few_shot_baseline3_format1 = Prompt(
    "Few-Shot Vanilla 3 Format 1", baseline_prompts[2], 1, shots=True
)
few_shot_baseline3_format2 = Prompt(
    "Few-Shot Vanilla 3 Format 2", baseline_prompts[2], 2, shots=True
)

few_shot_CoT1_format1 = Prompt(
    "Few-Shot CoT 1 Format 1", baseline_prompts[0], 1, CoT=True, shots=True
)

few_shot_CoT2_format1 = Prompt(
    "Few-Shot CoT 2 Format 1", baseline_prompts[1], 1, CoT=True, shots=True
)

few_shot_CoT3_format1 = Prompt(
    "Few-Shot CoT 3 Format 1", baseline_prompts[2], 1, CoT=True, shots=True
)

few_shot_CoT1_format2 = Prompt(
    "Few-Shot CoT 1 Format 2", baseline_prompts[0], 2, CoT=True, shots=True
)

few_shot_CoT2_format2 = Prompt(
    "Few-Shot CoT 2 Format 2", baseline_prompts[1], 2, CoT=True, shots=True
)

few_shot_CoT3_format2 = Prompt(
    "Few-Shot CoT 3 Format 2", baseline_prompts[2], 2, CoT=True, shots=True
)


prompts_to_test = [
    # zero_shot_baseline1_format1,
    # zero_shot_baseline1_format2,
    # zero_shot_baseline2_format1,
    # zero_shot_baseline2_format2,
    # zero_shot_baseline3_format1,
    # zero_shot_baseline3_format2,
    # few_shot_baseline1_format1,
    # few_shot_baseline1_format2
    # zero_shot_CoT
    # zero_shot_CoT1_format1,
    # zero_shot_CoT1_format2,
    # zero_shot_CoT2_format1,
    # zero_shot_CoT2_format2,
    # zero_shot_CoT3_format1,
    # zero_shot_CoT3_format2,
    # few_shot_baseline1_format1,
    # few_shot_baseline1_format2,
    # few_shot_baseline2_format1,
    # few_shot_baseline2_format2,
    # few_shot_baseline3_format1,
    # few_shot_baseline3_format2,
    # few_shot_CoT1_format1,
    # few_shot_CoT1_format2,
    # few_shot_CoT2_format1,
    # few_shot_CoT2_format2,
    # few_shot_CoT3_format1,
    # few_shot_CoT3_format2,
]

dataset = "mmlu"  # mmlu or gsm8k
config_name = None  # main if gs8k, None if mmlu
split = "test"
# model = "gpt-4-1106-preview"
model = "gpt-3.5-turbo"
examples = 2800  # number of examples to test
start = 0  # start index for dataset
log_interval = 200  # log interval for creatings jsons of results by query
max_toks = 700  # max tokens for query
rereading = False  # if true, will "reread" the question to the LM at query time
return_json = False
SEED = 42
temp = 0.5

def eval_prompts():
    eval = evaluate_prompts(
        prompts_to_test,
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
    d_path = f"data/benchmarking/eval_results_{current_datetime}.json"
    
    file_path = os.path.join(
        DataFolderPath,
        "experiments_output" + os.sep + d_path,
    )

    # Writing the dictionary to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(eval, json_file)

class Experiment:
    def run():
        eval_prompts()