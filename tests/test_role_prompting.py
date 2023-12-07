from src.prompt_systematic_review.role_prompting import (
    query_model,
    extract_numbers,
    evaluate_prompts,
    evaluate_gsm8k_response,
    evaluate_mmlu_response,
    response_to_dict,
)

import pytest
from dotenv import load_dotenv
import os
import openai


@pytest.fixture(scope="module")
def api_key():
    load_dotenv(dotenv_path="./.env")  # Adjust the path as needed
    return os.getenv("OPENAI_API_KEY")


@pytest.mark.API_test
def test_json_evaluation(api_key):
    openai.api_key = api_key
    prompts = [
        "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
        "You are a foolish high-school student. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
    ]

    model = "gpt-3.5-turbo-1106"
    dataset = "gsm8k"
    config_name = "main"
    split = "test"
    examples = 1

    evaluation_output = evaluate_prompts(
        prompts,
        dataset,
        config_name,
        split,
        model,
        examples,
    )

    # First JSON Object Test (Evaluation Summary)
    evaluation_summary = evaluation_output[0]
    for prompt, summary in evaluation_summary.items():
        assert isinstance(prompt, str)
        assert "correct" in summary and "total" in summary
        assert isinstance(summary["correct"], int)
        assert isinstance(summary["total"], int)

    # Second JSON Object Test (Detailed Responses)
    detailed_responses = evaluation_output[1]
    assert "dataset" in detailed_responses and isinstance(
        detailed_responses["dataset"], str
    )
    assert "config_name" in detailed_responses and isinstance(
        detailed_responses["config_name"], str
    )
    assert "split" in detailed_responses and isinstance(
        detailed_responses["split"], str
    )
    assert "model_name" in detailed_responses and isinstance(
        detailed_responses["model_name"], str
    )
    assert "examples" in detailed_responses and isinstance(
        detailed_responses["examples"], int
    )
    assert "total_input_tokens" in detailed_responses and isinstance(
        detailed_responses["total_input_tokens"], int
    )

    assert "total_output_tokens" in detailed_responses and isinstance(
        detailed_responses["total_output_tokens"], int
    )

    assert "total_wall_time" in detailed_responses and isinstance(
        detailed_responses["total_wall_time"], float
    )

    assert "calls" in detailed_responses and isinstance(
        detailed_responses["calls"], list
    )

    for call in detailed_responses["calls"]:
        assert "prompt" in call and isinstance(call["prompt"], str)
        assert "question" in call and isinstance(call["question"], str)
        assert "correct_answer" in call and isinstance(call["correct_answer"], str)
        assert "response" in call and isinstance(call["response"], dict)
        assert "marked_correct" in call and isinstance(call["marked_correct"], bool)
        assert "wall_time" in call and isinstance(call["wall_time"], float)

        # Test the structure of the 'response' object
        response = call["response"]
        assert "id" in response and isinstance(response["id"], str)
        assert "model" in response and isinstance(response["model"], str)
        assert "object" in response and isinstance(response["object"], str)
        assert "created" in response and isinstance(response["created"], int)
        assert "system_fingerprint" in response and isinstance(
            response["system_fingerprint"], str
        )
        assert "choices" in response and isinstance(response["choices"], list)
        assert "usage" in response and isinstance(response["usage"], dict)


@pytest.mark.API_test
def test_json_output(api_key):
    openai.api_key = api_key
    prompts = [
        "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
        "You are a foolish high-school student. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
    ]

    model = "gpt-3.5-turbo-1106"

    response = query_model(
        prompts[0],
        "What is 4 + 4?",
        model_name=model,
        output_tokens=150,
    )

    response_dict = response_to_dict(response)

    # Check the main keys
    assert "id" in response_dict
    assert "model" in response_dict
    assert "object" in response_dict
    assert "created" in response_dict
    assert "system_fingerprint" in response_dict
    assert "choices" in response_dict
    assert "usage" in response_dict

    # Check the types of the main keys
    assert isinstance(response_dict["id"], str)
    assert isinstance(response_dict["model"], str)
    assert isinstance(response_dict["object"], str)
    assert isinstance(response_dict["created"], int)
    assert isinstance(response_dict["system_fingerprint"], str)
    assert isinstance(response_dict["choices"], list)
    assert isinstance(response_dict["usage"], dict)

    # Check the structure and types of the 'choices' key
    assert len(response_dict["choices"]) > 0
    for choice in response_dict["choices"]:
        assert "finish_reason" in choice
        assert "index" in choice
        assert "message" in choice
        assert isinstance(choice["finish_reason"], str)
        assert isinstance(choice["index"], int)
        assert isinstance(choice["message"], dict)
        assert "content" in choice["message"]
        assert "role" in choice["message"]
        assert isinstance(choice["message"]["content"], str)
        assert isinstance(choice["message"]["role"], str)

    # Check the structure and types of the 'usage' key
    assert "completion_tokens" in response_dict["usage"]
    assert "prompt_tokens" in response_dict["usage"]
    assert "total_tokens" in response_dict["usage"]
    assert isinstance(response_dict["usage"]["completion_tokens"], int)
    assert isinstance(response_dict["usage"]["prompt_tokens"], int)
    assert isinstance(response_dict["usage"]["total_tokens"], int)


@pytest.mark.API_test
def test_query_model(api_key):
    openai.api_key = api_key
    prompt = "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible."
    question = "What is 4 + 4?"
    model_name = "gpt-3.5-turbo-1106"
    output_tokens = 150
    response = query_model(prompt, question, model_name, output_tokens)
    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices[0].message.content) > 0
    assert "8" in response.choices[0].message.content


@pytest.mark.API_test
def test_evaluate_prompts(api_key):
    openai.api_key = api_key
    prompts = [
        "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
        "You are a foolish high-school student. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
    ]

    dataset = "gsm8k"
    config_name = "main"
    split = "test"
    model = "gpt-3.5-turbo-1106"
    examples = 1

    eval, _ = evaluate_prompts(
        prompts,
        dataset,
        config_name,
        split,
        model,
        examples,
    )
    assert isinstance(eval, dict)
    assert len(eval) == 2


# def test_evaluate_mmlu_response():
#     class Response:
#         def __init__(self, content):
#             self.message = Message(content)

#     class Message:
#         def __init__(self, content):
#             self.content = content

#     response = Response(
#         "Hey! Here's how I got to the answer, \n First I did step 1, then step 2 and finally step 9999: \nQuailman was the first ever superhero created my Marvel."
#     )
#     correct_answer = "A"
#     answer_dict = {
#         "A": "Quailman was the first ever superhero created my Marvel",
#         "B": "Falconman was the first ever Marvel created superhero",
#         "C": "Quailwoman was the first ever superhero created by Marvel",
#         "D": "No superheroes were created by Marvel",
#     }

#     assert evaluate_mmlu_response(response, correct_answer, answer_dict) == True

#     response = Response(
#         "Hey! Here's how I got to the answer, \n First I did step 1, then step 2 and finally step 9999: \nQuailman was the last ever superhero created my Marvel."
#     )
#     assert evaluate_mmlu_response(response, correct_answer, answer_dict) == False


def test_evaluate_gsm8k_response():
    class Response:
        def __init__(self, content):
            self.message = Message(content)

    class Message:
        def __init__(self, content):
            self.content = content

    response = Response(
        "Hey! Here's how I got to the answer, \n First I did step 1, then step 2 and finally step 9999: \n####8"
    )
    correct_answer = "####8"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response(
        "Hey! Here's how I got to the answer, \n First I did step 1, then step 2 and finally step 9999: \r\n####8\t\n"
    )
    correct_answer = "####8"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####9"
    assert evaluate_gsm8k_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_gsm8k_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_gsm8k_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_gsm8k_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_gsm8k_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_gsm8k_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_gsm8k_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_gsm8k_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_gsm8k_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_gsm8k_response(response, correct_answer) == True


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
