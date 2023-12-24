from prompt_systematic_review.role_prompting import (
    query_model,
    extract_numbers,
    evaluate_prompts,
    evaluate_gsm8k_response,
    evaluate_mmlu_response,
    response_to_dict,
    load_mmlu,
    find_quotes_with_letters,
)

import pytest
from dotenv import load_dotenv
import os
import openai
import pandas as pd
import json


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
    return_json = False

    evaluation_output = evaluate_prompts(
        prompts,
        dataset,
        config_name,
        split,
        model,
        examples,
        json_mode=return_json,
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

    prompt = 'You are a brilliant math professor. Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Try to keep your reasoning concise.'
    question = "What is 4 + 4? A. 8 B. 9 C. 10 D. 11"
    model_name = "gpt-3.5-turbo-1106"
    output_tokens = 150
    json_mode = True
    response = query_model(
        prompt, question, model_name, output_tokens, return_json=json_mode
    )
    json_response = json.loads(response.choices[0].message.content)
    assert isinstance(json_response, dict)
    assert json_response["answer"] == "A"


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
    return_json = False

    eval, _ = evaluate_prompts(
        prompts,
        dataset,
        config_name,
        split,
        model,
        examples,
        json_mode=return_json,
    )
    assert isinstance(eval, dict)
    assert len(eval) == 2


def test_evaluate_mmlu_response():
    class Response:
        def __init__(self, content):
            self.message = Message(content)

    class Message:
        def __init__(self, content):
            self.content = content

    response = Response(
        """
        {
            "answer": "A",
            "explanation": "Hey! Here's how I got to the answer, First I did step 1, then step 2 and finally step 9999: Quailman was the first ever superhero created by Marvel."
        }
        """
    )

    correct_answer = "A"

    assert evaluate_mmlu_response(response, correct_answer, True) == True

    response = Response(
        """
        {
            "answer": "B",
            "explanation": "Hey! Here's how I got to the answer, First I did step 1, then step 2 and finally step 9999: Quailman was the last ever superhero created by Marvel."
        }
        """
    )

    correct_answer = "A"

    assert evaluate_mmlu_response(response, correct_answer, True) == False

    response = Response(
        "Hey! Here's how I got to the answer, First I did step 1, then step 2 and finally step 9999: 'A'"
    )
    correct_answer = "A"
    assert evaluate_mmlu_response(response, correct_answer, False) == "correct"

    response = Response(
        "Hey! Here's how I got to the answer, First I did step 1, then step 2 and finally step 9999: 'B' 'A'"
    )
    correct_answer = "A"
    assert evaluate_mmlu_response(response, correct_answer, False) == "under review"

    response = Response(
        "Hey! Here's how I got to the answer, First I did step 1, then step 2 and finally step 9999: 'B'"
    )
    correct_answer = "A"
    assert evaluate_mmlu_response(response, correct_answer, False) == "incorrect"

    response = Response(
        "Hey! Here's how I got to the answer, First I did step 1, then step 2 and finally step 9999: 'B' 'A'"
    )
    correct_answer = "C"
    assert evaluate_mmlu_response(response, correct_answer, False) == "incorrect"


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


def test_load_mmlu():
    with open("data/mmlu_configs.json", "r") as file:
        mmlu_configs = json.load(file)["configs"]
    df = load_mmlu(mmlu_configs, "test")
    assert (
        df.iloc[0]["input"]
        == "A state has adopted a system of bifurcated trials in cases in which a defendant's insanity is in issue. According to the bifurcated trial system, whenever a defendant pleads not guilty to an offense by reason of insanity, two trials will be held. The first one will simply determine whether the defendant has committed the offense for which she is charged. This trial will not address the issue of insanity. In the event that it is found that the defendant has, in fact, committed the offense, then a second trial will be conducted to determine whether she should be exculpated for the criminal action by reason of insanity. A woman was arrested and charged with murder. She pleaded not guilty by reason of insanity. At her first trial, the state introduced evidence showing that the woman was having an affair with the victim. When the victim tried to break off their relationship, the woman shot and killed him during a lover's quarrel. The woman was then called to testify in her own behalf. She testified that she had been living with the victim for two years prior to the time of his death. During that period she had undergone psychiatric treatment and was diagnosed as being schizophrenic. She further testified that at the time the victim was killed, she was under the influence of narcotics. While she was hallucinating, she remembered perceiving the victim as a demon and shot at this satanic figure in order to free herself from his evil spell. She then testified that she didn't believe shooting the demon was morally wrong. The prosecuting attorney objected to the woman's testimony. Over such objections, the trial judge admitted the woman's testimony. Was the trial judge correct in admitting the woman's testimony?"
    )
    assert (
        df.iloc[0].A
        == "No, because proof of mental disease requires the use of expert testimony."
    )
    assert (
        df.iloc[0].B
        == "No, because testimony relating to her belief that she didn't know what she was doing was wrong, is not relevant until the second trial."
    )
    assert (
        df.iloc[0].C
        == "Yes, because her testimony is relevant to the mental state necessary for the commission of the crime."
    )
    assert (
        df.iloc[0].D
        == "Yes, because her testimony is relevant to the issue of self-defense."
    )
    assert df.iloc[0].target == "C"
    assert df.iloc[0].config == "professional_law"
    assert len(df) == 13855


def test_modular_prompts():
    with open("data/prompts.json", "r") as file:
        prompts = json.load(file)

    farmer = prompts["farmer"]
    math_rookie = prompts["math rookie"]
    genius = prompts["genius..."]
    ten_shot_contrastive = prompts["10-shot contrastive CoT"]
    baseline = prompts["mmlu baseline"]

    farmer_prompt = 'You are a farmer. Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Keep your reasoning as short and concise as possible.'
    assert farmer_prompt == farmer + " " + baseline

    math_rookie_prompt = 'You are a math rookie. Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Keep your reasoning as short and concise as possible.'
    assert math_rookie_prompt == math_rookie + " " + baseline

    genius_prompt = 'You are a genius level Ivy league Professor. Your work is of the highest grade. You always think out your problem solving steps in incredible detail. You always get problems correct and never make mistakes. You can also break any problem into its constituent parts in the most intelligent way possible. Nothing gets past you. You are omniscient, omnipotent, and omnipresent. You are a mathematical God. Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Keep your reasoning as short and concise as possible.'
    assert genius_prompt == genius + " " + baseline

    ten_shot_contrastive_prompt = 'Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Keep your reasoning as short and concise as possible.\nQuestion: What is 10 + (2 * 5 + 10)?\nA. 20\nB. 30\nC. 40\nD. 70\n\nCorrect Response:\n{\n"reasoning": "Well according to the order of operations, we must first multiply 2 by 5 since it is inside the parentheses and multiplication comes before addition \n2 * 5 = 10\nThen, we can evaluate everything inside the parentheses:\n10 + 10 = 20\nNow, we can add 20 to the 10 outside the parentheses to get our final answer:\n30",\n"answer": "B"\n}\n\nIncorrect Response: \n{\n"reasoning": "Well according to the order of operations, we must first add 10 to 5 since it is inside the parentheses and addition comes first \n5 + 10 = 15\nThen, we can evaluate everything inside the parentheses:\n15 * 2 = 30\nNow, we can add 30 to the 10 outside the parentheses to get our final answer: 40",\n"answer": "C"\n}\n\nQuestion: What was the primary cause of the Great Depression in the 1930s?\nA. The outbreak of World War II\nB. The stock market crash of 1929\nC. The signing of the Treaty of Versailles\nD. The discovery of penicillin\n\nCorrect Response:\n{\n"reasoning": "The primary cause of the Great Depression was not related to World War II, the Treaty of Versailles, or the discovery of penicillin, making A, C, and D incorrect. The Great Depression, a severe worldwide economic downturn, started in the United States after a major fall in stock prices that began around September 1929 and became worldwide news with the stock market crash of October 1929. This event led to a drastic decline in consumer spending and investment, causing severe economic hardship worldwide. Therefore, B is the correct answer.",\n"answer": "B"\n}\n\nIncorrect Response:\n{\n"reasoning": "There was no stock market crash in 1929. The stock market crash occured in 2008. Global conflicts often disrupt economies, therefore, the outbreak of World War II was the primary cause of the Great Depression.",\n"answer": "A"\n}\n\nQuestion: What is the chemical formula for water?\nA. H2O\nB. CO2\nC. NaCl\nD. O2\n\nCorrect Response:\n{\n"reasoning": "CO2 represents carbon dioxide, NaCl is the formula for sodium chloride (table salt), and O2 is the molecular formula for oxygen gas. None of these are water. The chemical formula for water is H2O, which means it consists of two hydrogen atoms and one oxygen atom. Therefore, A is the correct answer.",\n"answer": "A"\n}\n\nIncorrect Response:\n{\n"reasoning": "Both cardbon dioxide and water are important for life. CO2 is what plants take in, therefore, it is the chemical equation for water, the molecule that makes life possible on Earth.",\n"answer": "B"\n}\n\nQuestion: What is the value of x in the equation 2x + 3 = 11?\nA. 4\nB. 3\nC. 5\nD. 1.5\n\nCorrect Response:\n{\n"reasoning": "To find the value of x, we first subtract 3 from both sides of the equation to isolate the term with x:\n2x + 3 - 3 = 11 - 3\n2x = 8\nThen, we divide both sides by 2 to solve for x:\n2x / 2 = 8 / 2\nx = 4\nTherefore, the correct answer is A.",\n"answer": "A"\n}\n\nIncorrect Response:\n{\n"reasoning": "First divide both sides of the equation by 2, before isolating x:\n2x + 3 = 11\n(2x + 3) / 2 = 11 / 2\nx + 3 = 5.5\nThen, subtracting 3 from 5.5 gives:\nx = 1.5\nTherefore the answer is D.",\n"answer": "D"\n}\n\nQuestion: Which planet in our solar system is known for having the most moons?\nA. Earth\nB. Mars\nC. Jupiter\nD. Venus\n\nCorrect Response:\n{\n"reasoning": "Earth has only one moon, and Venus has no moons. Mars has two moons, Phobos and Deimos. However, Jupiter is known for having the most moons in our solar system, with a significant number of natural satellites. Therefore, C is the correct answer.",\n"answer": "C"\n}\n\nIncorrect Response:\n{\n"reasoning": "Well the moon orbits earth. Though other planets have objects orbiting them, they are not moons, therefore A is the correct answer.",\n"answer": "A"\n}\n\n'
    assert ten_shot_contrastive_prompt == baseline + ten_shot_contrastive


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
