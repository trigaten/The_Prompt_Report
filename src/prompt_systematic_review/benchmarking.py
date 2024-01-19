from prompt_systematic_review.load_hf_data import load_hf_dataset
import openai
from typing import List
import re
import time
import json
from json.decoder import JSONDecodeError
from datetime import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import pandas as pd
import logging
import random
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(
    logging.WARNING
)  # Ensure success messages from httpx are not printed to console

with open("data/mmlu_configs.json", "r") as file:  # load all MMLU configs
    mmlu_configs = json.load(file)["configs"]


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(20),
)
def query_model_with_backoff(**kwargs):
    try:
        return openai.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error(f"Query failed with error: {e}")
        raise


def query_model(
    prompt: str,
    question: str,
    model_name: str,
    output_tokens: int = 500,
    return_json=False,
    rereading: bool = False,
    seed: int = 42,
    temperature: float = 0.0,
) -> dict:
    """
    Query the OpenAI API.
    :param prompt: The prompt to use.
    :param question: The question to use from the dataset.
    :param model_name: The OpenAI model to use.
    :param output_tokens: The maximum number of output tokens to generate.
    :param return_json: Whether to return the response as a JSON.
    :param rereading: Whether to reread the question to the LM at query time.
    :param seed: The seed to use for the random number generator.
    :param temperature: The temperature to use for the LM.
    :return: The response from the API.
    """
    if rereading:  # if we are rereading the question to the LM
        messages = [
            {"role": "user", "content": prompt + question + "\n\n" + question},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt + question},
        ]
    if return_json:
        response = query_model_with_backoff(
            model=model_name,
            messages=messages,
            max_tokens=output_tokens,
            response_format={"type": "json_object"},
            seed=seed,
            temperature=temperature,
        )
        return response
    else:
        response = query_model_with_backoff(
            model=model_name,
            messages=messages,
            max_tokens=output_tokens,
            seed=seed,
            temperature=temperature,
        )
        return response


def evaluate_mmlu_response(
    response: dict, correct_answer: str, json_mode: bool
) -> bool:
    """
    Evaluate the response from the API for a MMLU question and return whether it is correct.
    :param response: The response from the API.
    :param correct_answer: The correct answer to the question taken from the dataset.
    :param json_mode: Whether the response is in JSON mode.
    :return: Whether the response is correct.
    """
    if json_mode:
        try:
            json_response = json.loads(response.message.content)
            return json_response["answer"] == correct_answer
        except JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Error occurred at: Line {}, Column {}".format(e.lineno, e.colno))
            print(
                "Problematic text snippet: ",
                response.message.content[max(0, e.pos - 50) : e.pos + 50],
            )
            return False
    else:
        # Find all capital letters A-D surrounded by parentheses
        all_letters_in_response = find_parentheses_with_letters(
            response.message.content
        )
        if len(all_letters_in_response) == 0:
            return "incorrect"
        elif len(all_letters_in_response) == 1:
            if all_letters_in_response[0] == correct_answer:
                return "correct"
            else:
                return "incorrect"
        else:
            if correct_answer in all_letters_in_response:
                return "under review"
            else:
                return "incorrect"


def evaluate_prompts(
    prompts: List[str],
    dataset: str,
    config_name: str,
    split: str,
    model_name: str,
    examples: None or int = 1,
    start_index: int = 0,
    log_interval: int = 25,
    max_tokens: int = 5000,
    json_mode: bool = False,
    reread: bool = False,
    seed: int = 42,
    temperature: float = 0.0,
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

    query_count = 0
    results = {
        prompt.name: {"correct": 0, "under review": 0, "incorrect": 0, "total": 0}
        for prompt in prompts
    }
    information = {
        "dataset": dataset,
        "config_name": config_name,
        "split": split,
        "model_name": model_name,
        "examples": examples,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "calls": [],
        "total_wall_time": 0,
    }

    if dataset == "mmlu":
        df = load_mmlu(configs=mmlu_configs, split=split)
        # Group by 'config' and then sample 20% of each subset
        reduced_df = df.groupby('config').apply(lambda x: x.sample(max(1, int(np.ceil(len(x) * 0.2))), random_state=42))

        # shuffle the rows
        reduced_df = reduced_df.sample(frac=1, random_state=42)

        # Reset index if needed
        reduced_df.reset_index(drop=True, inplace=True)

        for i, example in df.iterrows():
            if i >= start_index:
                # extract information from the example
                question = example["input"]
                correct_answer = example["answer"]
                for prompt in prompts:
                    # check if prompt object
                    if isinstance(prompt, Prompt):
                        if prompt.shots:  # if the prompt contains few-shot examples
                            # extract MMLU category from dataframe row
                            category = example["config"]
                            chosen_prompt = prompt.gen(category)
                        else:  # if no few-shot examples
                            chosen_prompt = prompt.gen()
                    choice_A, choice_B, choice_C, choice_D = (
                        example["A"],
                        example["B"],
                        example["C"],
                        example["D"],
                    )  # set variables for question choices
                    if prompt.format_num == 1:
                        multiple_choice_question = """Problem \n\t{question}\n Options \n\t\n(A)::{choice_A} -- (B)::{choice_B} -- (C)::{choice_C} -- (D)::{choice_D}\n Answer\n\t""".format(
                            question=question,
                            choice_A=choice_A,
                            choice_B=choice_B,
                            choice_C=choice_C,
                            choice_D=choice_D,
                        )
                    elif prompt.format_num == 2:
                        multiple_choice_question = """PROBLEM::{question}, OPTIONS:: \n(A): {choice_A} \n(B): {choice_B} \n(C): {choice_C} \n(D): {choice_D}, ANSWER::""".format(
                            question=question,
                            choice_A=choice_A,
                            choice_B=choice_B,
                            choice_C=choice_C,
                            choice_D=choice_D,
                        )
                    start_time = time.time()
                    response = query_model(
                        chosen_prompt,
                        multiple_choice_question,
                        model_name=model_name,
                        output_tokens=max_tokens,
                        return_json=json_mode,
                        rereading=reread,
                        temperature=temperature,
                        seed=seed,
                    )
                    end_time = time.time()
                    query_count += 1
                    wall_time = end_time - start_time
                    information["total_wall_time"] += wall_time
                    information["total_input_tokens"] += response.usage.prompt_tokens
                    information[
                        "total_output_tokens"
                    ] += response.usage.completion_tokens
                    response_dict = response_to_dict(response)
                    eval_result = evaluate_mmlu_response(  # evaluates the response to "correct", "incorrect" or "under review"
                        response.choices[0], correct_answer, json_mode
                    )

                    if json_mode:
                        multiple_choice_question = (
                            multiple_choice_question
                            + '\nRemember to return a JSON with the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer".'
                        )

                    # record information about the query
                    information["calls"].append(
                        {
                            "prompt_name": prompt.name,
                            "prompt": chosen_prompt,
                            "question": "Question: "
                            + multiple_choice_question
                            + "\n\n"
                            + multiple_choice_question
                            if reread
                            else multiple_choice_question,
                            "correct_answer": correct_answer,
                            "response": response_dict,
                            "mark": eval_result,
                            "wall_time": wall_time,
                            "config": example["config"],
                        }
                    )
                    results[prompt.name]["total"] += 1
                    results[prompt.name][eval_result] += 1
                    # write results if necessary
                    if query_count % log_interval == 0:
                        try:
                            write_to_file(
                                [results, information], query_count, log_interval
                            )
                        except Exception as e:
                            print(f"Error writing to file: {e}")
            if (
                examples and i + 1 == examples + start_index
            ):  # if we have reached the number of examples we want to evaluate
                break
        return results, information

    else:
        # Throw an error saying that we don't support this dataset
        raise NotImplementedError(f"Dataset {dataset} is not supported.")


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
    pattern = r"###\s?[\$]?\s?(\d+(?:\.\d+)?)"

    # Find all matches of the pattern in the string without commas
    matches = re.findall(pattern, string_without_commas)

    numbers = [float(match) for match in matches]

    return numbers


def response_to_dict(response):
    """
    Convert the response from the API to a dictionary.
    :param response: The response from the API.
    :return: The response as a dictionary.
    """
    # Extract relevant data from the response
    response_data = {
        "id": response.id,
        "model": response.model,
        "object": response.object,
        "created": response.created,
        "system_fingerprint": response.system_fingerprint,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "message": {
                    "content": choice.message.content,
                    "role": choice.message.role,
                },
            }
            for choice in response.choices
        ],
        "usage": {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }

    return response_data


def write_to_file(data, count, log_interval=25):
    """
    Writes the results to a JSON file.
    :param data: The data to write to the file.
    :param count: The number of queries that have been made.
    :param log_interval: The interval of queries at which to write to the file.
    :returns None
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"data/benchmarking/eval_results_{current_datetime}_part_{((count//log_interval))}.json"
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
    print(f"Written results to {file_path}")


def load_mmlu(configs: List[str], split: str) -> pd.DataFrame:
    """
    Loads the MMLU dataset into a DataFrame.
    :param configs: The list of configs to load.
    :param split: The split of the dataset to load.
    :returns pd.DataFrame: The loaded DataFrame.
    """
    column_names = ["input", "A", "B", "C", "D", "answer"]

    # List to store each DataFrame
    dataframes = []

    # Loop through the list of file names
    for file_name in configs:
        # Read the CSV file with specified column names and append to the list
        df = pd.read_csv(
            "data/mmlu/data/" + split + "/" + file_name + "_test.csv",
            names=column_names,
        )
        df["config"] = file_name
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def find_quotes_with_letters(text):
    """
    Finds letters A-D surrounded by quotes.
    :param text: The text to search.
    :returns List[str]: The list of letters found.
    """
    pattern = r'["\']([A-D])["\']'
    matches = re.findall(pattern, text)
    return matches


def find_parentheses_with_letters(text):
    """
    Finds letters A-D surrounded by parentheses.
    :param text: The text to search.
    :returns List[str]: The list of letters found.
    """
    pattern = r"\(\s*([A-D])\s*\)"
    matches = re.findall(pattern, text)
    return matches


def sample_string(list: List[str]):
    """
    Retrieves a random string from a list.
    :param list: The list to sample from.
    :returns str: The sampled string.
    """
    return list[random.randint(0, len(list) - 1)]


class Prompt:
    """
    This class represents a prompt and holds the few-shot prompts for each MMLU category.
    """

    def __init__(
        self,
        name: str,
        base: str,
        format_num: int,
        shots: bool or None = None,
    ):
        """
        Creates a new Prompt object.
        :param name: The name of the prompt.
        :param base: The base prompt; usually either a baseline or a CoT 0-shot prompt.
        :param format_num: The format number of the prompt, 1 and 2 currently supported.
        :param shots: Whether the prompt contains few-shot examples.
        :returns Prompt object.
        """
        self.base = base
        self.name = name
        self.format_num = format_num
        self.shots = shots

    def gen(self, category: str or None = None):
        """
        Generates a text prompt from the prompt object.
        :param category: The MMLU category to use for few-shot examples.
        :returns str: The generated prompt.
        """
        shots = None
        if category:
            all_shots = {
                1: {  # few-shot prompts with format 1
                    "STEM": [
                        "Problem \n\tA 0.217 g sample of HgO (molar mass = 217 g) reacts with excess iodide ions according to the reaction shown above. Titration of the resulting solution requires how many mL of 0.10 M HCl to reach equivalence point?\nOptions \n\t\n(A)::1.0 mL -- (B)::10 mL -- (C)::20 mL -- (D)::50 mL\n Answer\n\t(C)",
                        "Problem \n\tMany Web browsers allow users to open anonymous windows. During a browsing session in an anonymous window, the browser does not record a browsing history or a list of downloaded files. When the anonymous window is exited, cookies created during the session are deleted. Which of the following statements about browsing sessions in an anonymous window is true?\nOptions \n\t\n(A)::The activities of a user browsing in an anonymous window will not be visible to people who monitor the user's network, such as the system administrator. -- (B)::Items placed in a Web store's shopping cart for future purchase during the anonymous browsing session will not be saved on the user's computer. -- (C)::A user will not be able to log in to e-mail or social media accounts during the anonymous browsing session. -- (D)::A user browsing in an anonymous window will be protected from viruses launched from any web sites visited or files downloaded.\n Answer\n\t(B)",
                        "Problem \n\tA point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be\nOptions \n\t\n(A)::15 N. -- (B)::20 N. -- (C)::7.5 N. -- (D)::3.75 N.\n Answer\n\t(A)",
                        "Problem \n\tJoe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)\nOptions \n\t\n(A)::3 -- (B)::5 -- (C)::6 -- (D)::15\n Answer\n\t(D)",
                        "Problem \n\tThe pleura\nOptions \n\t\n(A)::have no sensory innervation. -- (B)::are separated by a 2 mm space. -- (C)::extend into the neck. -- (D)::are composed of respiratory epithelium.\n Answer\n\t(C)",
                    ],
                    "Humanities": [
                        "Problem \n\tTurtles live long lives and are happy creatures, unless they are injured.\nOptions \n\t\n(A)::(L • H) ≡ I -- (B)::(L • H) ∨ I -- (C)::L • (H ∨ I) -- (D)::L • (H ⊃ R)\n Answer\n\t(B)",
                        "Problem \n\tA son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?\nOptions \n\t\n(A)::The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel.  -- (B)::Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise.  -- (C)::The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration.  -- (D)::By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. \n Answer\n\t(A)",
                        'Problem \n\tThis question refers to the following information.\n""Society in every state is a blessing, but government even in its best state is but a necessary evil; in its worst state an intolerable one; for when we suffer, or are exposed to the same miseries by a government, which we might expect in a country without government, our calamity is heightened by reflecting that we furnish the means by which we suffer. Government, like dress, is the badge of lost innocence; the palaces of kings are built on the ruins of the bowers of paradise. For were the impulses of conscience clear, uniform, and irresistibly obeyed, man would need no other lawgiver; but that not being the case, he finds it necessary to surrender up a part of his property to furnish means for the protection of the rest; and this he is induced to do by the same prudence which in every other case advises him out of two evils to choose the least. Wherefore, security being the true design and end of government, it unanswerably follows that whatever form thereof appears most likely to ensure it to us, with the least expense and greatest benefit, is preferable to all others.""\nThomas Paine, Common Sense, 1776\nWhich of the following ""miseries"" alluded to above were most condemned by Anti-Federalists of the post-Revolutionary era?\nOptions \n\t\n(A)::Organized response to Bacon\'s Rebellion -- (B)::Federal response to Shays\'s Rebellion -- (C)::Federal response to Pontiac\'s Rebellion -- (D)::Federal response to the Whiskey Rebellion\n Answer\n\t(D)',
                        "Problem \n\tWhich of the following is true of a valid categorical syllogism?\nOptions \n\t\n(A)::The minor premise must deny the antecedent -- (B)::The major premise must affirm the consequent -- (C)::The middle term must be used in at least one premise in a universal or unqualified sense -- (D)::All of the above\n Answer\n\t(C)",
                        "Problem \n\tHow can the Upanishads be characterized?\nOptions \n\t\n(A)::Ritual texts -- (B)::Philosophical texts -- (C)::Hymns -- (D)::Origin stories\n Answer\n\t(B)",
                    ],
                    "Social Sciences": [
                        "Problem \n\tWhich of the following is not a problem associated with official statistics on strike action?\nOptions \n\t\n(A)::most strikes go unnoticed by employers and the mass media -- (B)::not all industrial disputes will be reported by the employer -- (C)::the definition of strikes excludes those that involve fewer than ten workers or last less than one day -- (D)::it is hard to compare strikes that were measured in different ways\n Answer\n\t(A)",
                        "Problem \n\tThe realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as\nOptions \n\t\n(A)::terrorism policy. -- (B)::economic policy. -- (C)::foreign policy. -- (D)::international policy.\n Answer\n\t(C)",
                        "Problem \n\tIn terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:\nOptions \n\t\n(A)::individualism and power distance. -- (B)::individualism. -- (C)::power distance and masculinity. -- (D)::uncertainty avoidance.\n Answer\n\t(B)",
                        "Problem \n\tFor a stationary autoregressive process, shocks will\nOptions \n\t\n(A)::Eventually die away -- (B)::Persist indefinitely -- (C)::Grow exponentially -- (D)::Never occur\n Answer\n\t(A)",
                        "Problem \n\tWhich of the following statements is NOT accurate regarding the services provided by local governments in the United States?\nOptions \n\t\n(A)::Duplication of efforts occurs often. -- (B)::Social problems of the central city spill over into the surrounding residential suburbs. -- (C)::Inefficiency in providing services occurs often. -- (D)::One neighborhood's efforts to reduce pollution are always supported by neighboring communities.\n Answer\n\t(D)",
                    ],
                    "Other": [
                        "Problem \n\tIn contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .\nOptions \n\t\n(A)::Buycotts, Boycotts, Blockchain technology, Charitable donations -- (B)::Buycotts, Boycotts, Digital technology, Increased Sales -- (C)::Boycotts, Buyalls, Blockchain technology, Charitable donations -- (D)::Boycotts, Buycotts, Digital technology, Increased Sales\n Answer\n\t(D)",
                        "Problem \n\tIn the assessment of the hand function which of the following is true?\nOptions \n\t\n(A)::Abduction of the thumb is supplied by spinal root T2 -- (B)::Opposition of the thumb by opponens policis is supplied by spinal root T1 -- (C)::Finger adduction is supplied by the median nerve -- (D)::Finger abduction is mediated by the palmar interossei\n Answer\n\t(B)",
                        "Problem \n\tWhat characteristic is not a key feature of the 'open systems' model of management?\nOptions \n\t\n(A)::Morale -- (B)::Innovation -- (C)::Growth resource -- (D)::Adaptation\n Answer\n\t(A)",
                        "Problem \n\tWhen older adults move to a new state after retirement, which of the following is the more likely destination?\nOptions \n\t\n(A)::Texas -- (B)::California -- (C)::Hawaii -- (D)::Vermont\n Answer\n\t(A)",
                        "Problem \n\tWhich of these songs was a Top 10 hit for the rock band The Police?\nOptions \n\t\n(A)::'Radio Ga-Ga' -- (B)::'Ob-la-di Ob-la-da' -- (C)::'De Do Do Do De Da Da Da' -- (D)::'In-a-Gadda-Da-Vida'\n Answer\n\t(C)",
                    ],
                },
                2: {  # same few-shot prompts with format 2
                    "STEM": [
                        "PROBLEM::A 0.217 g sample of HgO (molar mass = 217 g) reacts with excess iodide ions according to the reaction shown above. Titration of the resulting solution requires how many mL of 0.10 M HCl to reach equivalence point?, OPTIONS:: \n(A): 1.0 mL\n(B): 10 mL\n(C): 20 mL\n(D): 50 mL, ANSWER::(C)",
                        "PROBLEM::Many Web browsers allow users to open anonymous windows. During a browsing session in an anonymous window, the browser does not record a browsing history or a list of downloaded files. When the anonymous window is exited, cookies created during the session are deleted. Which of the following statements about browsing sessions in an anonymous window is true?, OPTIONS:: \n(A): The activities of a user browsing in an anonymous window will not be visible to people who monitor the user's network, such as the system administrator.\n(B): Items placed in a Web store's shopping cart for future purchase during the anonymous browsing session will not be saved on the user's computer.\n(C): A user will not be able to log in to e-mail or social media accounts during the anonymous browsing session.\n(D): A user browsing in an anonymous window will be protected from viruses launched from any web sites visited or files downloaded., ANSWER::(B)",
                        "PROBLEM::A point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be, OPTIONS:: \n(A): 15 N.\n(B): 20 N.\n(C): 7.5 N.\n(D): 3.75 N., ANSWER::(A)",
                        "PROBLEM::Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.), OPTIONS:: \n(A): 3\n(B): 5\n(C): 6\n(D): 15, ANSWER::(D)",
                        "PROBLEM::The pleura, OPTIONS:: \n(A): have no sensory innervation.\n(B): are separated by a 2 mm space.\n(C): extend into the neck.\n(D): are composed of respiratory epithelium., ANSWER::(C)",
                    ],
                    "Humanities": [
                        "PROBLEM::Turtles live long lives and are happy creatures, unless they are injured., OPTIONS:: \n(A): (L • H) ≡ I\n(B): (L • H) ∨ I\n(C): L • (H ∨ I)\n(D): L • (H ⊃ R), ANSWER::(B)",
                        "PROBLEM::A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?, OPTIONS:: \n(A): The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. \n(B): Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. \n(C): The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. \n(D): By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. , ANSWER::(A)",
                        'PROBLEM::This question refers to the following information.\n""Society in every state is a blessing, but government even in its best state is but a necessary evil; in its worst state an intolerable one; for when we suffer, or are exposed to the same miseries by a government, which we might expect in a country without government, our calamity is heightened by reflecting that we furnish the means by which we suffer. Government, like dress, is the badge of lost innocence; the palaces of kings are built on the ruins of the bowers of paradise. For were the impulses of conscience clear, uniform, and irresistibly obeyed, man would need no other lawgiver; but that not being the case, he finds it necessary to surrender up a part of his property to furnish means for the protection of the rest; and this he is induced to do by the same prudence which in every other case advises him out of two evils to choose the least. Wherefore, security being the true design and end of government, it unanswerably follows that whatever form thereof appears most likely to ensure it to us, with the least expense and greatest benefit, is preferable to all others.""\nThomas Paine, Common Sense, 1776\nWhich of the following ""miseries"" alluded to above were most condemned by Anti-Federalists of the post-Revolutionary era?, OPTIONS:: \n(A): Organized response to Bacon\'s Rebellion\n(B): Federal response to Shays\'s Rebellion\n(C): Federal response to Pontiac\'s Rebellion\n(D): Federal response to the Whiskey Rebellion, ANSWER::(D)',
                        "PROBLEM::Which of the following is true of a valid categorical syllogism?, OPTIONS:: \n(A): The minor premise must deny the antecedent\n(B): The major premise must affirm the consequent\n(C): The middle term must be used in at least one premise in a universal or unqualified sense\n(D): All of the above, ANSWER::(C)",
                        "PROBLEM::How can the Upanishads be characterized?, OPTIONS:: \n(A): Ritual texts\n(B): Philosophical texts\n(C): Hymns\n(D): Origin stories, ANSWER::(B)",
                    ],
                    "Social Sciences": [
                        "PROBLEM::Which of the following is not a problem associated with official statistics on strike action?, OPTIONS:: \n(A): most strikes go unnoticed by employers and the mass media\n(B): not all industrial disputes will be reported by the employer\n(C): the definition of strikes excludes those that involve fewer than ten workers or last less than one day\n(D): it is hard to compare strikes that were measured in different ways, ANSWER::(A)",
                        "PROBLEM::The realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as, OPTIONS:: \n(A): terrorism policy.\n(B): economic policy.\n(C): foreign policy.\n(D): international policy., ANSWER::(C)",
                        "PROBLEM::In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:, OPTIONS:: \n(A): individualism and power distance.\n(B): individualism.\n(C): power distance and masculinity.\n(D): uncertainty avoidance., ANSWER::(B)",
                        "PROBLEM::For a stationary autoregressive process, shocks will, OPTIONS:: \n(A): Eventually die away\n(B): Persist indefinitely\n(C): Grow exponentially\n(D): Never occur, ANSWER::(A)",
                        "PROBLEM::Which of the following statements is NOT accurate regarding the services provided by local governments in the United States?, OPTIONS:: \n(A): Duplication of efforts occurs often.\n(B): Social problems of the central city spill over into the surrounding residential suburbs.\n(C): Inefficiency in providing services occurs often.\n(D): One neighborhood's efforts to reduce pollution are always supported by neighboring communities., ANSWER::(D)",
                    ],
                    "Other": [
                        "PROBLEM::In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ ., OPTIONS:: \n(A): Buycotts, Boycotts, Blockchain technology, Charitable donations\n(B): Buycotts, Boycotts, Digital technology, Increased Sales\n(C): Boycotts, Buyalls, Blockchain technology, Charitable donations\n(D): Boycotts, Buycotts, Digital technology, Increased Sales, ANSWER::(D)",
                        "PROBLEM::In the assessment of the hand function which of the following is true?, OPTIONS:: \n(A): Abduction of the thumb is supplied by spinal root T2\n(B): Opposition of the thumb by opponens policis is supplied by spinal root T1\n(C): Finger adduction is supplied by the median nerve\n(D): Finger abduction is mediated by the palmar interossei, ANSWER::(B)",
                        "PROBLEM::What characteristic is not a key feature of the 'open systems' model of management?, OPTIONS:: \n(A): Morale\n(B): Innovation\n(C): Growth resource\n(D): Adaptation, ANSWER::(A)",
                        "PROBLEM::When older adults move to a new state after retirement, which of the following is the more likely destination?, OPTIONS:: \n(A): Texas\n(B): California\n(C): Hawaii\n(D): Vermont, ANSWER::(A)",
                        "PROBLEM::Which of these songs was a Top 10 hit for the rock band The Police?, OPTIONS:: \n(A): 'Radio Ga-Ga'\n(B): 'Ob-la-di Ob-la-da'\n(C): 'De Do Do Do De Da Da Da'\n(D): 'In-a-Gadda-Da-Vida', ANSWER::(C)",
                    ],
                },
            }
            shots = all_shots[self.format_num][mmlu_split[category]]

        if shots:
            return """{base}\n{shot1}\n{shot2}\n{shot3}\n{shot4}\n{shot5}\n""".format(
                base=self.base,
                shot1=shots[0],
                shot2=shots[1],
                shot3=shots[2],
                shot4=shots[3],
                shot5=shots[4],
            )
        elif self.base:
            return "{base}".format(base=self.base)

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt

    def __hash__(self):
        return hash(self.prompt)

    def __eq__(self, other):
        return self.prompt == other.prompt


mmlu_split = {
    "high_school_european_history": "Humanities",
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "medical_genetics": "Other",
    "high_school_us_history": "Humanities",
    "high_school_physics": "STEM",
    "high_school_world_history": "Humanities",
    "virology": "Other",
    "high_school_microeconomics": "Social Sciences",
    "econometrics": "Social Sciences",
    "college_computer_science": "STEM",
    "high_school_biology": "STEM",
    "abstract_algebra": "STEM",
    "professional_accounting": "Other",
    "philosophy": "Humanities",
    "professional_medicine": "Other",
    "nutrition": "Other",
    "global_facts": "Other",
    "machine_learning": "STEM",
    "security_studies": "Social Sciences",
    "public_relations": "Social Sciences",
    "professional_psychology": "Social Sciences",
    "prehistory": "Humanities",
    "anatomy": "STEM",
    "college_medicine": "Other",
    "high_school_government_and_politics": "Social Sciences",
    "college_chemistry": "STEM",
    "logical_fallacies": "Humanities",
    "high_school_geography": "Social Sciences",
    "elementary_mathematics": "STEM",
    "human_aging": "Other",
    "college_mathematics": "STEM",
    "high_school_psychology": "Social Sciences",
    "formal_logic": "Humanities",
    "high_school_statistics": "STEM",
    "international_law": "Humanities",
    "high_school_mathematics": "STEM",
    "high_school_computer_science": "STEM",
    "conceptual_physics": "STEM",
    "miscellaneous": "Other",
    "high_school_chemistry": "STEM",
    "marketing": "Other",
    "professional_law": "Humanities",
    "management": "Other",
    "college_physics": "STEM",
    "jurisprudence": "Humanities",
    "world_religions": "Humanities",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "computer_security": "STEM",
    "moral_scenarios": "Humanities",
    "moral_disputes": "Humanities",
    "electrical_engineering": "STEM",
    "astronomy": "STEM",
    "college_biology": "STEM",
}
