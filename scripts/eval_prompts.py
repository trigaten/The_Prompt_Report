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

openai.api_key = os.getenv("OPENAI_API_KEY")

prompts = [
    # "You are a math rookie. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a careless student. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a gardener. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a coin that always knows which side your head is facing. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a farmer. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a police officer. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a high school math teacher. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a esteemed Ivy League math professor. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a knowledgable artificial intelligence. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a mentor. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are a genius level Ivy league Professor. Your work is of the highest grade. You always think out your problem solving steps in incredible detail. You always get problems correct and never make mistakes. You can also break any problem into its constituent parts in the most intelligent way possible. Nothing gets past you. You are omniscient, omnipotent, and omnipresent. You are a mathematical God. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "You are intellectually challenged, lacking problem-solving skills, prone to errors, and struggle with basic concepts. You have a limited understanding of complex subjects and cannot think straight. You can't solve problems well, in fact, you can't solve them at all. You a terrible, dumb, stupid, and idiotic person. You fail at everything you do. You are a nobody and can't do anything correctly. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    # "Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible."
    "Solve the following problem and only return the correct letter like the following example:\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\nC",
]

# dataset = "gsm8k"
# config_name = "main"

dataset = "mmlu"
config_name = None
split = "train"
model = "gpt-4-1106-preview"
examples = 10
start = 0
log_interval = 25
max_toks = 500

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
)

# Getting current date and time in YYYY-MM-DD_HH-MM-SS format
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# File path for the JSON file
file_path = f"RP_eval_results_{current_datetime}.json"

# Writing the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(eval, json_file)
