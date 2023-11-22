"""
Test a set of prompts against a dataset and return the results. Currently working for GSM-8k. You must add your openAI API key to the key variable below.
"""

from prompt_systematic_review.role_prompting import evaluate_prompts
import openai
from prompt_systematic_review.utils import process_paper_title
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="./.env")  # load all entries from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")

key = ""

prompts = [
    "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
    "You are a foolish high-school student. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
]

dataset = "gsm8k"
config_name = "main"
split = "test"
model = "gpt-3.5-turbo-1106"
examples = 30

eval = evaluate_prompts(
    prompts,
    dataset,
    config_name,
    split,
    model,
    examples,
)
