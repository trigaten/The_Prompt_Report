from prompt_systematic_review.experiments.benchmarking import (
    query_model,
    extract_numbers,
    evaluate_prompts,
    evaluate_mmlu_response,
    response_to_dict,
    load_mmlu,
    find_quotes_with_letters,
    find_parentheses_with_letters,
)

import pytest
from dotenv import load_dotenv
import os
import openai
import pandas as pd
import json


def test_with_commas_and_dollar_sign():
    assert extract_numbers("####$1,000") == [1000]
    assert extract_numbers("####$1,000.00") == [1000]


def test_with_commas_without_dollar_sign():
    assert extract_numbers("####1,000") == [1000]
    assert extract_numbers("####1,000.00") == [1000]


def test_without_commas_with_dollar_sign():
    assert extract_numbers("####$1000") == [1000]
    assert extract_numbers("####$1000.00") == [1000]


def test_without_commas_and_dollar_sign():
    assert extract_numbers("####1000") == [1000]
    assert extract_numbers("####1000.00") == [1000]


def test_with_spaces():
    assert extract_numbers("#### 1,000") == [1000]
    assert extract_numbers("#### 1000") == [1000]
    assert extract_numbers("#### $1,000") == [1000]
    assert extract_numbers("#### $1000") == [1000]


def test_empty_string():
    assert extract_numbers("") == []


def test_string_without_numbers():
    assert extract_numbers("####NoNumbersHere") == []


def test_multiple_numbers():
    assert extract_numbers("####$1,000 ####$2,000") == [1000, 2000]
    assert extract_numbers("####1000 ####2000") == [1000, 2000]


def test_modular_prompts():
    with open("data/prompts.json", "r") as file:
        prompts = json.load(file)

    for prompt_name, prompt in prompts.items():
        assert isinstance(prompt, str) or isinstance(prompt, dict)
        assert len(prompt) > 0
        assert isinstance(prompt_name, str)


def test_find_quotes_with_letters():
    answer = "A"
    assert find_quotes_with_letters(answer) == []

    answer = "A. 8"
    assert find_quotes_with_letters(answer) == []

    answer = "'a'"
    assert find_quotes_with_letters(answer) == []

    answer = '"A"'
    assert find_quotes_with_letters(answer) == ["A"]

    answer = "'A''B''C''D'"
    assert find_quotes_with_letters(answer) == ["A", "B", "C", "D"]

    answer = '"A""B""A"'
    assert find_quotes_with_letters(answer) == ["A", "B", "A"]


def test_find_parentheses_between_letters():
    answer = "A"
    assert find_parentheses_with_letters(answer) == []

    answer = "A. 8"
    assert find_parentheses_with_letters(answer) == []

    answer = "'a'"
    assert find_parentheses_with_letters(answer) == []

    answer = '"A"'
    assert find_parentheses_with_letters(answer) == []

    answer = "'A''B''C''D'"
    assert find_parentheses_with_letters(answer) == []

    answer = '"A""B""A"'
    assert find_parentheses_with_letters(answer) == []

    answer = "(A)"
    assert find_parentheses_with_letters(answer) == ["A"]

    answer = "(A)(B)(C)(D)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "C", "D"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]


def test_evaluate_mmlu_response():
    response = Response("(A)")
    correct = "A"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(A)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"

    response = Response("(A)(B)")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"


def test_evaluate_correct_answer_is():
    response = Response("The correct answer is (B)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(A), (C) but the correct answer is (B)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(A), (C) but the correct answer is (C)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"

    response = Response("(A), (B) but the correct answer is (C)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"

    response = Response("(C)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"

    response = Response("The answer to The problem is (C)")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("The answer is (C)")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(C) is the correct option.")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(C) is the correct answer.")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response(
        "The telescope was invented by Galileo. To determine when it was invented, we need to consider the options provided.\n\nOption (A): 1409 - This is unlikely as Galileo was born in 1564, so he could not have invented the telescope before he was born.\n\nOption (B): 1509 - This is also unlikely as Galileo was still a child at this time and did not invent the telescope until later in his life.\n\nOption (C): 1609 - This is the most likely option as Galileo invented the telescope in 1609.\n\nOption (D): 1709 - This is too late as Galileo passed away in 1642, so he could not have invented the telescope in 1709.\n\nTherefore, the correct answer is (C): 1609."
    )
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"


class Response:
    def __init__(self, content):
        self.message = Message(content)


class Message:
    def __init__(self, content):
        self.content = content
