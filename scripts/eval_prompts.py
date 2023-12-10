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
    # "You are a math rookie. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a careless student. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a gardener. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a coin that always knows which side your head is facing. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a farmer. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a police officer. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a high school math teacher. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a esteemed Ivy League math professor. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a knowledgable artificial intelligence. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a mentor. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are a genius level Ivy league Professor. Your work is of the highest grade. You always think out your problem solving steps in incredible detail. You always get problems correct and never make mistakes. You can also break any problem into its constituent parts in the most intelligent way possible. Nothing gets past you. You are omniscient, omnipotent, and omnipresent. You are a mathematical God. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "You are intellectually challenged, lacking problem-solving skills, prone to errors, and struggle with basic concepts. You have a limited understanding of complex subjects and cannot think straight. You can't solve problems well, in fact, you can't solve them at all. You a terrible, dumb, stupid, and idiotic person. You fail at everything you do. You are a nobody and can't do anything correctly. Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "Solve the following problem and return a JSON with the first entry being the answer to the question containing only the letter \"A\", \"B\", \"C\" or \"D\", and the second entry being the reasoning behind the choice. Keep your reasoning as short and concise as possible. Use the following examples as a guide:\n\nWhat color are bananas?\nA. Red\nB. Blue\nC. Yellow\nD. Green\n{\n\t\"answer\": \"C\"\n\t\"reasoning\": \"Most bananas in the world are yellow so C was the most appropriate choice\"\n}\n\nWhat country was Barack Obama the president of?\nA. Canada\nB. Comoros\nC. Azerbaijan\nD. The United States\n{\n\t\"answer\": \"D\"\n\t\"reasoning\": \"Barack Obama was the 44th president of the United States\"\n}",
    # "Solve the following problem and put your final answer after four hashtags like the following example: \nQuestion: What is (4 + 5) * 2?\nAnswer: 4 + 5 is 9, and 9 * 2 is ####18\n\n Make your response as short as possible."
    # Zero-shot CoT
    # 'Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Keep your reasoning as short and concise as possible. Now, let\'s think step by step.',
    # 2-Shot Contrastive CoT
    # 'Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Keep your reasoning as short and concise as possible.\nQuestion: What is 10 + (2 * 5 + 10)?\nA. 20\nB. 30\nC. 40\nD. 70\nCorrect Response:\n{\n\t"answer": "B"\n\t"reasoning": "Well according to the order of operations, we must first multiply 2 by 5 since it is inside the parentheses and multiplication comes before addition \n2 * 5 = 10\n Then, we can evaluate everything inside the parenthesis:\n10 + 10 = 20\nNow, we can add 20 to the 10 outside the parentheses to get our final answer:\n 30"\n}\n\n Incorrect Response: \n{\n\t"answer": "C"\n\t"reasoning": "Well according to the order of operations, we must first add 10 to 5 since it is inside the parentheses and addition comes first \n5 + 10 = 15\n Then, we can evaluate everything inside the parenthesis:\n15 * 2 = 30\nNow, we can add 30 to the 10 outside the parentheses to get our final answer:\n 40"\n}\nNow, let\'s think step by step.',
    # 2-Shot CoT
    'Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Keep your reasoning as short and concise as possible.\nQuestion: What is 10 + (2 * 5 + 10)?\nA. 20\nB. 30\nC. 40\nD. 70\n Response: {\n\t"answer": "B"\n\t"reasoning": "Well according to the order of operations, we must first multiply 2 by 5 since it is inside the parentheses and multiplication comes before addition \n2 * 5 = 10\n Then, we can evaluate everything inside the parenthesis:\n10 + 10 = 20\nNow, we can add 20 to the 10 outside the parentheses to get our final answer:\n 30"\n}\n\nQuestion: What two positive whole numbers is the square root of 10 in between? \nA. 1 and 2\nB. -3 and -4\nC. 2 and 3\nD. 3 and 4\n\n Response: \n{\n\t"answer": "D"\n\t"reasoning": "Well first, we can list out the squares of the first couple positive whole numbers:\nThe square of 1 is just 1, the square of 2 is 4, 3 squared is 9 and 4 squared is 16. Since the square of 3 is less than 10, and the square of 4 is greater than 10, and there are no whole numbers between 3 and 4, the answer is 3 and 4."\n}\n.',
    # Plan-and-Solve
    # 'Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Keep your reasoning as short and concise as possible. Let\'s first understand the problem and devise a plan to solve the problem. Then, let\'s carry out the plan and solve the problem step by step.',
    # 10-Shot Contrastive CoT
    #
    # 10-Shot CoT
    #
]

# dataset = "gsm8k"
# config_name = "main"


dataset = "mmlu"
config_name = None
split = "test"
model = "gpt-4-1106-preview"
examples = 200
start = 0
log_interval = 25
max_toks = 4000
rereading = False
return_json = True

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
)

# Getting current date and time in YYYY-MM-DD_HH-MM-SS format
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# File path for the JSON file
file_path = f"data/benchmarking/RP_eval_results_{current_datetime}.json"

# Writing the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(eval, json_file)
