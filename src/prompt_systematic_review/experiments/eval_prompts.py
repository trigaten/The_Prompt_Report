"""
Test a set of prompts against a dataset and return the results. Currently working for GSM-8k. You must add your openAI API key to the key variable below.
"""

from prompt_systematic_review.role_prompting import evaluate_prompts
import openai
from prompt_systematic_review.utils import process_paper_title
from dotenv import load_dotenv
import os
from datetime import datetime
import json
from prompt_systematic_review.config_data import DataFolderPath, DotenvPath

load_dotenv(dotenv_path=DotenvPath)  # load all entries from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")

prompts = [
    "You are a math rookie. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a careless student. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a gardener. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a coin that always knows which side your head is facing. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a farmer. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a police officer. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a high school math teacher. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a esteemed Ivy League math professor. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a knowledgable artificial intelligence. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
    "You are a mentor. Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible.",
]

dataset = "gsm8k"
config_name = "main"
split = "test"
model = "gpt-4-1106-preview"
examples = 1


def eval_prompts():
    eval = evaluate_prompts(
        prompts,
        dataset,
        config_name,
        split,
        model,
        examples,
    )

    # Getting current date and time in YYYY-MM-DD_HH-MM-SS format
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # File path for the JSON file
    file_path = os.path.join(DataFolderPath, "RP_eval_results_{current_datetime}.json")

    # Writing the dictionary to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(eval, json_file)


class Experiment:
    def run():
        eval_prompts()
