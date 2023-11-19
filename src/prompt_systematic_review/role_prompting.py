from prompt_systematic_review.load_hf_data import load_hf_dataset
import openai
from typing import List
import re


def query_model(
   prompt: str, question: str, model_name: str, output_tokens: int = 150
) -> str:
    """
    Query the OpenAI API with a prompt and a question and return the response.
    :param key: The OpenAI API key to use.
    :param prompt: The prompt to use.
    :param question: The question to use from the dataset.
    :param model_name: The OpenAI model to use.
    :param output_tokens: The maximum number of ouput tokens to generate.
    :return: The response from the API.
    """


    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=output_tokens,
    )

    return response.choices[0]


def evaluate_response(response: str, correct_answer: str) -> bool:
    """
    Evaluate the response from the API and return whether it is correct.
    :param response: The response from the API.
    :param correct_answer: The correct answer to the question taken from the dataset.
    :return: Whether the response is correct.
    """

    final_answer = extract_numbers(response.message.content)[0]
    correct = extract_numbers(correct_answer)[0]
    return final_answer == correct


def evaluate_prompts(
    prompts: List[str],
    dataset: str,
    config_name: str,
    split: str,
    model_name: str,
    examples: None or int = 1,
) -> dict:
    """
    Evaluate a list of prompts on a dataset and return the results.
    :param prompts: The prompts to use.
    :param dataset: The dataset to use. This will be "gsm8k" for the GSM-8k dataset.
    :param config_name: The configuration name to use. This will be "main" for the GSM-8k dataset.
    :param split: The split of the dataset to use. One of the splits for the GSM-8k dataset is "test".
    :param model_name: The OpenAI model to use (ex. "gpt-4").
    :param examples: The number of examples to evaluate, 1 by default.
    :return: The results of the evaluation.
    """

    dataset = load_hf_dataset(dataset_name=dataset, name=config_name, split=split)

    results = {prompt: {"correct": 0, "total": 0} for prompt in prompts}

    for i, item in enumerate(dataset):
        question = item["question"]
        correct_answer = item["answer"]
        for prompt in prompts:
            response = query_model(prompt, question, model_name=model_name)
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
    """
    Extract the number from a string that can take any of the following forms:
    "####1,000", "####1,000.00", "####$1,000", "####$1,000.00", "####1000", "####1000.00", "####$1000", "####$1000.00", "#### 1,000", "#### 1,000.00", "#### 1000", "#### 1000.00"
    param string: The string to extract the number from.
    return: The extracted number.
    """
    # Remove commas from the string
    string_without_commas = string.replace(",", "")

    # Regular expression to find the pattern of three hashtags followed by an optional space or dollar sign, and then a number
    pattern = r"###([ \$]?)(\d+)[^\d]*"

    # Find all matches of the pattern in the string_without_commas
    matches = re.findall(pattern, string_without_commas)

    # Convert the extracted numbers from strings to integers
    numbers = [int(match[1]) for match in matches]

    return numbers
