from load_hf_data import load_hf_dataset
from openai import OpenAI
from typing import List
import re


def query_model(
    key: str, prompt: str, question: str, model_name: str, output_tokens: int = 150
) -> str:
    client = OpenAI(
        api_key=key,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=output_tokens,
    )

    return response.choices[0]


def evaluate_response(response: str, correct_answer: str) -> bool:
    print(response.message.content)
    final_answer = extract_numbers(response.message.content)[0]
    correct = extract_numbers(correct_answer)[0]
    return final_answer == correct


def evaluate_prompts(
    API_key: str,
    prompts: List[str],
    dataset: str,
    config_name: str,
    split: str,
    model_name: str,
    examples: None or int = 1,
) -> dict:
    dataset = load_hf_dataset(dataset_name=dataset, name=config_name, split=split)

    results = {prompt: {"correct": 0, "total": 0} for prompt in prompts}

    for i, item in enumerate(dataset):
        question = item["question"]
        correct_answer = item["answer"]
        print(f"Question: {question}\nAnswer: {correct_answer}\n")
        for prompt in prompts:
            response = query_model(API_key, prompt, question, model_name=model_name)
            is_correct = evaluate_response(response, correct_answer)
            results[prompt]["total"] += 1
            if is_correct:
                results[prompt]["correct"] += 1

        if examples and i >= examples:
            break

    for prompt, result in results.items():
        accuracy = result["correct"] / result["total"]
        print(f"Prompt: {prompt}\nAccuracy: {accuracy:.2f}\n")


def extract_numbers(string: str) -> List[int]:
    # Remove commas from the string
    string_without_commas = string.replace(",", "")

    # Regular expression to find the pattern of three hashtags followed by an optional space or dollar sign, and then a number
    pattern = r"###([ \$]?)(\d+)[^\d]*"

    # Find all matches of the pattern in the string_without_commas
    matches = re.findall(pattern, string_without_commas)

    # Convert the extracted numbers from strings to integers
    numbers = [int(match[1]) for match in matches]

    return numbers


key = ""

prompts = [
    "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
    "You are a foolish high-school student. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
]

dataset = "gsm8k"
model = "gpt-4"

eval = evaluate_prompts(
    key,
    prompts,
    dataset="gsm8k",
    config_name="main",
    split="test",
    model_name=model,
    examples=10,
)
