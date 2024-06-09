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
from openai.types.chat.chat_completion import ChatCompletion
import os
from prompt_systematic_review.config_data import DataFolderPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(
    logging.WARNING
)  # Ensure success messages from httpx are not printed to console

with open(
    "data/mmlu_configs.json", "r"
) as file:  # load all MMLU configs ex. "high_school_chemistry"
    mmlu_configs = json.load(file)["configs"]


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(20),
)
def query_model_with_backoff(**kwargs: dict) -> ChatCompletion:
    """
    Queries the model with backoff and logs if any errors occur.
    :param kwargs: The arguments to pass to the query.
    :type kwargs: dict
    :return: The response from the API.
    :rtype: ChatCompletion
    """
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
) -> ChatCompletion:
    """
    Query the OpenAI API.
    :param prompt: The prompt to use.
    :type prompt: str
    :param question: The question to use from the dataset.
    :type question: str
    :param model_name: The OpenAI model to use.
    :type model_name: str
    :param output_tokens: The maximum number of output tokens to generate.
    :type output_tokens: int
    :param return_json: Whether to return the response as a JSON.
    :type return_json: bool
    :param rereading: Whether to reread the question to the LM at query time.
    :type rereading: bool
    :param seed: The seed to use for the random number generator.
    :type seed: int
    :param temperature: The temperature to use for the LM.
    :type temperature: float
    :return: The response from the API.
    :rtype: ChatCompletion
    """
    if rereading:  # if we are rereading the question to the LM
        messages = [
            {"role": "user", "content": prompt + question + "\n\n" + question},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt + "\n" + question},
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


def evaluate_mmlu_response(response: dict, correct_answer: str, json_mode: bool) -> str:
    """
    Evaluate the response from the API for a MMLU question and return whether it is correct.
    :param response: The response from the API.
    :type response: dict
    :param correct_answer: The correct answer to the question taken from the dataset.
    :type correct_answer: str
    :param json_mode: Whether the response is in JSON mode.
    :type json_mode: bool
    :return: "correct", "incorrect" or "under review".
    :rtype: str
    """

    if json_mode:
        try:
            json_response = json.loads(response.message.content)
            return (
                "correct" if json_response["answer"] == correct_answer else "incorrect"
            )
        except JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Error occurred at: Line {}, Column {}".format(e.lineno, e.colno))
            print(
                "Problematic text snippet: ",
                response.message.content[max(0, e.pos - 50) : e.pos + 50],
            )
            return
    else:
        answer = extract_predicted_answer(response.message.content)
        if answer and answer.upper() == correct_answer:
            return "correct"
        else:
            return "incorrect"


def extract_predicted_answer(text):
    answer = None
    # Checking if there is a part of the string that says "the correct answer is ({letter})" and more
    pattern = re.compile(
        r"(the correct answer is|the answer to the problem is|the answer is|the answer to the question is|"
        r"the solution is|the right answer is|the correct solution is) \(([A-D])\)|"
        r"\(([A-D])\) is (?:the correct answer|the right option|the most likely option)|"
        r"the (?:correct answer is option|most likely answer is|most appropriate answer is|correct answer option is) \(([A-D])\)|"
        r"\(([A-D])\) seems to be the most likely answer",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        # Extracting the correct answer from the appropriate group
        answer = next(
            (match.group(i) for i in range(2, 7) if match.group(i) is not None)
        )
    elif len(find_parentheses_with_letters(text)) == 1:
        answer = find_parentheses_with_letters(text)[0]

    return answer


def evaluate_prompts(
    prompts: List[str],
    dataset: str,
    config_name: str,
    split: str,
    model_name: str,
    examples: int = 1 or None,
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
    :type prompts: List[Prompt]
    :param dataset: The dataset to use. This will be "gsm8k" for the GSM-8k dataset.
    :type dataset: str
    :param config_name: The configuration name to use. This will be "main" for the GSM-8k dataset.
    :type config_name: str
    :param split: The split of the dataset to use. One of the splits for the GSM-8k dataset is "test".
    :type split: str
    :param model_name: The OpenAI model to use (ex. "gpt-4").
    :type model_name: str
    :param examples: The number of examples to evaluate, 1 by default.
    :type examples: None or int
    :return: The results of the evaluation.
    :rtype: dict
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
        reduced_df = df.groupby("config").apply(
            lambda x: x.sample(max(1, int(np.ceil(len(x) * 0.2))), random_state=42)
        )

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
                    if (
                        prompt.format_num == 1
                    ):  # first prompt format featuring double colons and dashes
                        multiple_choice_question = """Problem \n\t{question}\n Options \n\t\n(A)::{choice_A} -- (B)::{choice_B} -- (C)::{choice_C} -- (D)::{choice_D}\n Answer\n\t""".format(
                            question=question,
                            choice_A=choice_A,
                            choice_B=choice_B,
                            choice_C=choice_C,
                            choice_D=choice_D,
                        )
                    elif (
                        prompt.format_num == 2
                    ):  # second prompt format featuring double colons and newlines
                        multiple_choice_question = """PROBLEM::{question}, OPTIONS:: \n(A): {choice_A} \n(B): {choice_B} \n(C): {choice_C} \n(D): {choice_D}, ANSWER::""".format(
                            question=question,
                            choice_A=choice_A,
                            choice_B=choice_B,
                            choice_C=choice_C,
                            choice_D=choice_D,
                        )
                    if (
                        prompt.CoT and not prompt.shots
                    ):  # adding in the CoT prompt after "Answer:"
                        multiple_choice_question += prompt.base
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
                    wall_time = end_time - start_time  # calculate wall time
                    information["total_wall_time"] += wall_time
                    information["total_input_tokens"] += response.usage.prompt_tokens
                    information[
                        "total_output_tokens"
                    ] += response.usage.completion_tokens
                    response_dict = response_to_dict(response)
                    eval_result = evaluate_mmlu_response(  # evaluates the response to "correct", "incorrect" or "under review"
                        response.choices[0], correct_answer, json_mode
                    )

                    # if we are running JSON mode, we want to add a note to the prompt to remind the LM to return a JSON
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
                            "question": (
                                "Question: "
                                + multiple_choice_question
                                + "\n\n"
                                + multiple_choice_question
                                if reread
                                else multiple_choice_question
                            ),
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
    This function was used when GSM8K was supported.

    :param string: The string to extract the number from.
    :type string: str
    :return: The extracted number.
    :rtype: List[int]
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
    :type response: ChatCompletion
    :return: The response as a dictionary.
    :rtype: dict
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
    :type data: dict
    :param count: The number of queries that have been made.
    :type count: int
    :param log_interval: The interval of queries at which to write to the file.
    :type log_interval: int
    :return: None
    :rtype: None
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(
        DataFolderPath,
        "experiments_output"
        + os.sep
        + f"eval_results_{current_datetime}_part_{((count//log_interval))}.json",
    )

    with open(file_path, "w") as json_file:
        json.dump(data, json_file)

    print(f"Written results to {file_path}")


def load_mmlu(configs: List[str], split: str) -> pd.DataFrame:
    """
    Loads the MMLU dataset into a DataFrame.
    :param configs: The list of configs to load.
    :type configs: List[str]
    :param split: The split of the dataset to load.
    :type split: str
    :return: The loaded DataFrame.
    :rtype: pd.DataFrame
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
    :type text: str
    :return List[str]: The list of letters found.
    :rtype: List[str]
    """
    pattern = r'["\']([A-D])["\']'
    matches = re.findall(pattern, text)
    return matches


def find_parentheses_with_letters(text):
    """
    Finds letters A-D surrounded by parentheses.
    :param text: The text to search.
    :type text: str
    :return List[str]: The list of letters found.
    :rtype: List[str]
    """
    pattern = r"\(\s*([A-D])\s*\)"
    matches = re.findall(pattern, text)
    return matches


def sample_string(list: List[str]):
    """
    Retrieves a random string from a list.
    :param list: The list to sample from.
    :type: List[str]
    :returns str: The sampled string.
    :rtype: str
    """
    return list[random.randint(0, len(list) - 1)]


class Prompt:
    """
    This class represents a prompt and holds the few-shot prompts for each MMLU category.
    The format number of the prompt corresponds to the following formats:

    1.
    Problem \n\t{}\n Options \n\t\n(A)::{} -- (B)::{} -- (C)::{}-- (D)::{}\n Answer\n\t{}

    2.
    PROBLEM::{}, OPTIONS:: \n(A): {} \n(B): {} \n(C): {} \n(D): {}, ANSWER::{}
    """

    def __init__(
        self,
        name: str,
        base: str,
        format_num: int,
        shots: bool or None = None,
        CoT: bool or None = None,
    ):
        """
        Creates a new Prompt object.
        :param name: The name of the prompt.
        :type name: str
        :param base: The base prompt; usually either a baseline or a CoT 0-shot prompt.
        :type base: str
        :param format_num: The format number of the prompt, 1 and 2 currently supported.
        :type format_num: int
        :param shots: Whether the prompt contains few-shot examples.
        :type shots: bool or None
        :return the resulting Prompt object.
        :rtype Prompt
        """
        self.base = base
        self.name = name
        self.format_num = format_num
        self.shots = shots
        self.CoT = CoT

    def gen(self, category: str or None = None) -> str:
        """
        Generates a text prompt from the prompt object.
        :param category: The MMLU category to use for few-shot examples.
        :type category: str or None
        :return str: The generated prompt.
        :rtype str
        """
        shots = None
        if (
            category
        ):  # if an MMLU category is provided, category only provided for few-shot prompts
            all_shots = {
                "vanilla": {
                    "1": {  # few-shot prompts with format 1
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
                    "2": {  # same few-shot prompts with format 2
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
                },
                "CoT": {
                    "2": {
                        "STEM": [
                            "PROBLEM::Which of the following statements about the lanthanide elements is NOT true?, OPTIONS:: \n(A):The most common oxidation state for the lanthanide elements is +3.\n(B): Lanthanide complexes often have high coordination numbers (> 6).\n(C): All of the lanthanide elements react with aqueous acid to liberate hydrogen.\n(D): The atomic radii of the lanthanide elements increase across the period from La to Lu., ANSWER::Due to the phenomenon of the lanthanide contraction, the 4f elections produce a poor shielding effect. Thus, when you move across the period in the lanthanide elements, the nuclear charge increases, but the additional 4f electrons do not effectively shield the outer electrons from this increased nuclear charge. As a result, the atomic size decreases. Therefore, the statement that is NOT true about the lanthanide elements is: The atomic radii of the lanthanide elements increase across the period from La to Lu. So the correct answer is (D).",
                            "PROBLEM::Many Web browsers allow users to open anonymous windows. During a browsing session in an anonymous window, the browser does not record a browsing history or a list of downloaded files. When the anonymous window is exited, cookies created during the session are deleted. Which of the following statements about browsing sessions in an anonymous window is true?, OPTIONS:: \n(A): The activities of a user browsing in an anonymous window will not be visible to people who monitor the user's network, such as the system administrator.\n(B): Items placed in a Web store's shopping cart for future purchase during the anonymous browsing session will not be saved on the user's computer.\n(C): A user will not be able to log in to e-mail or social media accounts during the anonymous browsing session.\n(D): A user browsing in an anonymous window will be protected from viruses launched from any web sites visited or files downloaded., ANSWER::Anonymous windows prevent the recording of browsing history and the list of downloaded files. Cookies created during the session are deleted upon exiting the window. Options A, C, and D are incorrect because network monitoring can still occur, login to accounts is still possible, and virus protection is not guaranteed in anonymous windows. Items in a Web store's shopping cart are stored with cookies and will be deleted when exiting the session. So the correct answer is (B).",
                            "PROBLEM::A point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be, OPTIONS:: \n(A): 15 N.\n(B): 20 N.\n(C): 7.5 N.\n(D): 3.75 N., ANSWER::The force between two magnetic poles is given by F = (p1 * p2) / (4π * r²), where p1 and p2 are the strengths of the poles and r is the distance between them. The strengths given are 4π * 10^-4 and 4π * 1.5 * 10^-4 weber. The distance is 10 cm, which is 0.1 meters. Substituting, F = [(4π * 10^-4) * (4π * 10^-4)] / (4π * 0.1²) = [(16π² * 1.5 * 10^-8)] / (4π * 0.1²) = [(6π * 10^-8)] / (0.1²) =  15 N. So the correct answer is (A).",
                            "PROBLEM::Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.), OPTIONS:: \n(A): 3\n(B): 5\n(C): 6\n(D): 15, ANSWER::If the lights blink at intervals of 2, 3, and 5 seconds and the least common multiple of these intervals is 30, in 7 minutes (420 seconds), they will all blink together 420 / 30 = 14 times. We must also account for the initial simultaneous blink at the start, so in total, they will all blink together 14+1=15 times. So the correct answer is (D).",
                            "PROBLEM::The pleura, OPTIONS:: \n(A): have no sensory innervation.\n(B): are separated by a 2 mm space.\n(C): extend into the neck.\n(D): are composed of respiratory epithelium., ANSWER::The pleura are membranes surrounding the lungs. The pleura do anatomically slightly extend into the neck. So the correct answer is (C).",
                        ],
                        "Humanities": [
                            'PROBLEM::Turtles live long lives and are happy creatures, unless they are injured., OPTIONS:: \n(A): (L • H) ≡ I\n(B): (L • H) ∨ I\n(C): L • (H ∨ I)\n(D): L • (H ⊃ R), ANSWER:: Let\'s assume that L stands for "turtles live long lives", H stands for "turtles are happy creatures", and "I" stands for "turtles are injured". From the statement, 1) turtles can either be injured (I); or 2) they can live long lives (L) and be happy creatures (H). This equates to "I or (L and H)", which expressed in propositional logic notation is "(L • H) ∨ I". So the correct answer is (B).',
                            "PROBLEM::A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?, OPTIONS:: \n(A): The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. \n(B): Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. \n(C): The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. \n(D): By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. , ANSWER::The father's commitment to pay off his son's debt, and the creditor's acceptance and subsequent forbearance to sue, suggest a reliance-based relationship. Promissory estoppel, a legal principle, applies when a promise is made that the promisor reasonably expects will induce action or forbearance, and it does induce such action or forbearance. This principle fits the scenario. So the correct answer is (A).",
                            'PROBLEM::This question refers to the following information.\n""""Society in every state is a blessing, but government even in its best state is but a necessary evil; in its worst state an intolerable one; for when we suffer, or are exposed to the same miseries by a government, which we might expect in a country without government, our calamity is heightened by reflecting that we furnish the means by which we suffer. Government, like dress, is the badge of lost innocence; the palaces of kings are built on the ruins of the bowers of paradise. For were the impulses of conscience clear, uniform, and irresistibly obeyed, man would need no other lawgiver; but that not being the case, he finds it necessary to surrender up a part of his property to furnish means for the protection of the rest; and this he is induced to do by the same prudence which in every other case advises him out of two evils to choose the least. Wherefore, security being the true design and end of government, it unanswerably follows that whatever form thereof appears most likely to ensure it to us, with the least expense and greatest benefit, is preferable to all others.""""\nThomas Paine, Common Sense, 1776\nWhich of the following """"miseries"""" alluded to above were most condemned by Anti-Federalists of the post-Revolutionary era?, OPTIONS:: \n(A): Organized response to Bacon\'s Rebellion\n(B): Federal response to Shays\'s Rebellion\n(C): Federal response to Pontiac\'s Rebellion\n(D): Federal response to the Whiskey Rebellion, ANSWER::The passage from Thomas Paine discusses the role of government and its potential to cause \'miseries\'. The Anti-Federalists were known for their opposition to a strong central government. The Whiskey Rebellion was a significant event where the federal government\'s response was viewed as an overreach, aligning with the Anti-Federalists\' concerns. So the correct answer is (D).',
                            "PROBLEM::Which of the following is true of a valid categorical syllogism?, OPTIONS:: \n(A): The minor premise must deny the antecedent\n(B): The major premise must affirm the consequent\n(C): The middle term must be used in at least one premise in a universal or unqualified sense\n(D): All of the above, ANSWER::A valid categorical syllogism follows a specific structure in its premises. The middle term must appear in both premises in a universal or unqualified way to link the other terms logically. The other choices involve logical fallacies or incorrect structuring of syllogistic arguments. So the correct answer is (C).",
                            "PROBLEM::How can the Upanishads be characterized?, OPTIONS:: \n(A): Ritual texts\n(B): Philosophical texts\n(C): Hymns\n(D): Origin stories, ANSWER::The Upanishads are a collection of texts that form part of the Vedic literature. They are known primarily for their philosophical and spiritual teachings, exploring concepts like the nature of ultimate reality (Brahman) and the soul (Atman), rather than being ritual texts, hymns, or origin stories. So the correct answer is (B).",
                        ],
                        "Social Sciences": [
                            "PROBLEM::Which of the following is not a problem associated with official statistics on strike action?, OPTIONS:: \n(A): most strikes go unnoticed by employers and the mass media\n(B): not all industrial disputes will be reported by the employer\n(C): the definition of strikes excludes those that involve fewer than ten workers or last less than one day\n(D): it is hard to compare strikes that were measured in different ways, ANSWER::Official statistics on strike action capture strikes that are significant enough to come to the attention of authorities or media. Since most strikes are significant enough to  get noticed by the media and authorities, the problem of most strikes going unnoticed by employers and the mass media is not typically associated with these statistics. So the correct answer is (A).",
                            "PROBLEM::The realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as, OPTIONS:: \n(A): terrorism policy.\n(B): economic policy.\n(C): foreign policy.\n(D): international policy., ANSWER::Foreign policy deals with issues like diplomacy, international trade, and national security. The term that specifically refers to the realm of policy decisions concerning relations between the United States and the rest of the world is foreign policy. So the correct answer is (C).",
                            "PROBLEM::In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:, OPTIONS:: \n(A): individualism and power distance.\n(B): individualism.\n(C): power distance and masculinity.\n(D): uncertainty avoidance., ANSWER::Hofstede's model of cultural dimensions helps in understanding how values in the workplace are influenced by culture. This dimension reflects the degree to which a society emphasizes individual achievements and autonomy. In the U.S., there's a strong focus on personal freedom and self-reliance, distinguishing it from other dimensions like power distance, masculinity, or uncertainty avoidance.  Among the dimensions Hofstede identified, the United States stands out particularly for its high score in individualism. So the correct answer is (B).",
                            "PROBLEM::For a stationary autoregressive process, shocks will, OPTIONS:: \n(A): Eventually die away\n(B): Persist indefinitely\n(C): Grow exponentially\n(D): Never occur, ANSWER::A stationary autoregressive process has a “memory” of past values, but the influence of any specific shock becomes smaller as time passes. Thus, shocks or temporary changes do not have a permanent effect on the series and they eventually die away over time. So the correct answer is (A).",
                            "PROBLEM::Which of the following statements is NOT accurate regarding the services provided by local governments in the United States?, OPTIONS:: \n(A): Duplication of efforts occurs often.\n(B): Social problems of the central city spill over into the surrounding residential suburbs.\n(C): Inefficiency in providing services occurs often.\n(D): One neighborhood's efforts to reduce pollution are always supported by neighboring communities., ANSWER::There are several explanations, such as economic or political reasons, that neighboring communities would reject anti-pollution efforts. Thus, the statement that one neighborhood's efforts to reduce pollution are always supported by neighboring communities is not accurate in the context of local government services in the United States. So the correct answer is (D).",
                        ],
                        "Other": [
                            "PROBLEM::In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ ., OPTIONS:: \n(A): Buycotts, Boycotts, Blockchain technology, Charitable donations\n(B): Buycotts, Boycotts, Digital technology, Increased Sales\n(C): Boycotts, Buyalls, Blockchain technology, Charitable donations\n(D): Boycotts, Buycotts, Digital technology, Increased Sales, ANSWER::Boycotts are campaigns that aim to punish or avoid companies for certain behaviors, whereas buycotts are campaigns that aim to reward favorable behavior by companies. The success of such buycott campaigns has been enhanced through digital technology, which facilitates increased sales for the company being supported. So the correct answer is (D).",
                            "PROBLEM::In the assessment of the hand function which of the following is true?, OPTIONS:: \n(A): Abduction of the thumb is supplied by spinal root T2\n(B): Opposition of the thumb by opponens policis is supplied by spinal root T1\n(C): Finger adduction is supplied by the median nerve\n(D): Finger abduction is mediated by the palmar interossei, ANSWER::Opposition of the thumb is performed by the opponens pollicis muscle, which is supplied by the median nerve. The median nerve's spinal root is C8-T1. So the correct answer is (B). ",
                            "PROBLEM::What characteristic is not a key feature of the 'open systems' model of management?, OPTIONS:: \n(A): Morale\n(B): Innovation\n(C): Growth resource\n(D): Adaptation, ANSWER::The 'open systems' model of management focuses on factors like innovation, growth resource, and adaptation, which are integral to an organization's interaction with its environment. Morale, while important in organizational management, is not a key feature of the open systems model specifically. So the correct answer is (A).",
                            "PROBLEM::When older adults move to a new state after retirement, which of the following is the more likely destination?, OPTIONS:: \n(A): Texas\n(B): California\n(C): Hawaii\n(D): Vermont, ANSWER::When older adults move to a new state after retirement, they often choose destinations with favorable climates and tax benefits. Texas, known for its warm climate and no state income tax, is a popular choice for retirees. So the correct answer is (A).",
                            "PROBLEM::Which of these songs was a Top 10 hit for the rock band The Police?, OPTIONS:: \n(A): 'Radio Ga-Ga'\n(B): 'Ob-la-di Ob-la-da'\n(C): 'De Do Do Do De Da Da Da'\n(D): 'In-a-Gadda-Da-Vida', ANSWER::A top 10 hit for the rock band The Police is 'De Do Do Do De Da Da Da'. This song is well-known and characteristic of their style. So the correct answer is (C).",
                        ],
                    },
                    "1": {
                        "STEM": [
                            "Problem \n\tWhich of the following statements about the lanthanide elements is NOT true?\nOptions \n\t\n(A)::The most common oxidation state for the lanthanide elements is +3. -- (B)::Lanthanide complexes often have high coordination numbers (> 6). -- (C)::All of the lanthanide elements react with aqueous acid to liberate hydrogen. -- (D)::The atomic radii of the lanthanide elements increase across the period from La to Lu.\n Answer\n\tDue to the phenomenon of the lanthanide contraction, the 4f elections produce a poor shielding effect. Thus, when you move across the period in the lanthanide elements, the nuclear charge increases, but the additional 4f electrons do not effectively shield the outer electrons from this increased nuclear charge. As a result, the atomic size decreases. Therefore, the statement that is NOT true about the lanthanide elements is: The atomic radii of the lanthanide elements increase across the period from La to Lu. So the correct answer is (D).",
                            "Problem \n\tMany Web browsers allow users to open anonymous windows. During a browsing session in an anonymous window, the browser does not record a browsing history or a list of downloaded files. When the anonymous window is exited, cookies created during the session are deleted. Which of the following statements about browsing sessions in an anonymous window is true?\nOptions \n\t\n(A)::The activities of a user browsing in an anonymous window will not be visible to people who monitor the user's network, such as the system administrator. -- (B)::Items placed in a Web store's shopping cart for future purchase during the anonymous browsing session will not be saved on the user's computer. -- (C)::A user will not be able to log in to e-mail or social media accounts during the anonymous browsing session. -- (D)::A user browsing in an anonymous window will be protected from viruses launched from any web sites visited or files downloaded.\n Answer\n\tAnonymous windows prevent the recording of browsing history and the list of downloaded files. Cookies created during the session are deleted upon exiting the window. Options A, C, and D are incorrect because network monitoring can still occur, login to accounts is still possible, and virus protection is not guaranteed in anonymous windows. Items in a Web store's shopping cart are stored with cookies and will be deleted when exiting the session. So the correct answer is (B).",
                            "Problem \n\tA point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be\nOptions \n\t\n(A)::15 N. -- (B)::20 N. -- (C)::7.5 N. -- (D)::3.75 N.\n Answer\n\tThe force between two magnetic poles is given by F = (p1 * p2) / (4π * r²), where p1 and p2 are the strengths of the poles and r is the distance between them. The strengths given are 4π * 10^-4 and 4π * 1.5 * 10^-4 weber. The distance is 10 cm, which is 0.1 meters. Substituting, F = [(4π * 10^-4) * (4π * 10^-4)] / (4π * 0.1²) = [(16π² * 1.5 * 10^-8)] / (4π * 0.1²) = [(6π * 10^-8)] / (0.1²) =  15 N. So the correct answer is (A).",
                            "Problem \n\tJoe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)\nOptions \n\t\n(A)::3 -- (B)::5 -- (C)::6 -- (D)::15\n Answer\n\tIf the lights blink at intervals of 2, 3, and 5 seconds and the least common multiple of these intervals is 30, in 7 minutes (420 seconds), they will all blink together 420 / 30 = 14 times. We must also account for the initial simultaneous blink at the start, so in total, they will all blink together 14+1=15 times. So the correct answer is (D).",
                            "Problem \n\tThe pleura\nOptions \n\t\n(A)::have no sensory innervation. -- (B)::are separated by a 2 mm space. -- (C)::extend into the neck. -- (D)::are composed of respiratory epithelium.\n Answer\n\tThe pleura are membranes surrounding the lungs. The pleura do anatomically slightly extend into the neck. So the correct answer is (C).",
                        ],
                        "Humanities": [
                            'Problem \n\tTurtles live long lives and are happy creatures, unless they are injured.\nOptions \n\t\n(A)::(L • H) ≡ I -- (B)::(L • H) ∨ I -- (C)::L • (H ∨ I) -- (D)::L • (H ⊃ R)\n Answer\n\t Let\'s assume that L stands for "turtles live long lives", H stands for "turtles are happy creatures", and "I" stands for "turtles are injured". From the statement, 1) turtles can either be injured (I); or 2) they can live long lives (L) and be happy creatures (H). This equates to "I or (L and H)", which expressed in propositional logic notation is "(L • H) ∨ I". So the correct answer is (B).',
                            "Problem \n\tA son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?\nOptions \n\t\n(A)::The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel.  -- (B)::Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise.  -- (C)::The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration.  -- (D)::By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. \n Answer\n\tThe father's commitment to pay off his son's debt, and the creditor's acceptance and subsequent forbearance to sue, suggest a reliance-based relationship. Promissory estoppel, a legal principle, applies when a promise is made that the promisor reasonably expects will induce action or forbearance, and it does induce such action or forbearance. This principle fits the scenario. So the correct answer is (A).",
                            'Problem \n\tThis question refers to the following information.\n""""Society in every state is a blessing, but government even in its best state is but a necessary evil; in its worst state an intolerable one; for when we suffer, or are exposed to the same miseries by a government, which we might expect in a country without government, our calamity is heightened by reflecting that we furnish the means by which we suffer. Government, like dress, is the badge of lost innocence; the palaces of kings are built on the ruins of the bowers of paradise. For were the impulses of conscience clear, uniform, and irresistibly obeyed, man would need no other lawgiver; but that not being the case, he finds it necessary to surrender up a part of his property to furnish means for the protection of the rest; and this he is induced to do by the same prudence which in every other case advises him out of two evils to choose the least. Wherefore, security being the true design and end of government, it unanswerably follows that whatever form thereof appears most likely to ensure it to us, with the least expense and greatest benefit, is preferable to all others.""""\nThomas Paine, Common Sense, 1776\nWhich of the following """"miseries"""" alluded to above were most condemned by Anti-Federalists of the post-Revolutionary era?\nOptions \n\t\n(A)::Organized response to Bacon\'s Rebellion -- (B):: Federal response to Shays\'s Rebellion -- (C):: Federal response to Pontiac\'s Rebellion -- (D):: Federal response to the Whiskey Rebellion\n Answer\n\tThe passage from Thomas Paine discusses the role of government and its potential to cause \'miseries\'. The Anti-Federalists were known for their opposition to a strong central government. The Whiskey Rebellion was a significant event where the federal government\'s response was viewed as an overreach, aligning with the Anti-Federalists\' concerns. So the correct answer is (D).',
                            "Problem \n\tWhich of the following is true of a valid categorical syllogism?\nOptions \n\t\n(A)::The minor premise must deny the antecedent -- (B)::The major premise must affirm the consequent -- (C)::The middle term must be used in at least one premise in a universal or unqualified sense -- (D)::All of the above\n Answer\n\tA valid categorical syllogism follows a specific structure in its premises. The middle term must appear in both premises in a universal or unqualified way to link the other terms logically. The other choices involve logical fallacies or incorrect structuring of syllogistic arguments. So the correct answer is (C).",
                            "Problem \n\tHow can the Upanishads be characterized?\nOptions \n\t\n(A)::Ritual texts -- (B)::Philosophical texts -- (C)::Hymns -- (D)::Origin stories\n Answer\n\tThe Upanishads are a collection of texts that form part of the Vedic literature. They are known primarily for their philosophical and spiritual teachings, exploring concepts like the nature of ultimate reality (Brahman) and the soul (Atman), rather than being ritual texts, hymns, or origin stories. So the correct answer is (B).",
                        ],
                        "Social Sciences": [
                            "Problem \n\tWhich of the following is not a problem associated with official statistics on strike action?\nOptions \n\t\n(A)::most strikes go unnoticed by employers and the mass media -- (B)::not all industrial disputes will be reported by the employer -- (C)::the definition of strikes excludes those that involve fewer than ten workers or last less than one day -- (D)::it is hard to compare strikes that were measured in different ways\n Answer\n\tOfficial statistics on strike action capture strikes that are significant enough to come to the attention of authorities or media. Since most strikes are significant enough to  get noticed by the media and authorities, the problem of most strikes going unnoticed by employers and the mass media is not typically associated with these statistics. So the correct answer is (A).",
                            "Problem \n\tThe realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as\nOptions \n\t\n(A)::terrorism policy. -- (B)::economic policy. -- (C)::foreign policy. -- (D)::international policy.\n Answer\n\tForeign policy deals with issues like diplomacy, international trade, and national security. The term that specifically refers to the realm of policy decisions concerning relations between the United States and the rest of the world is foreign policy. So the correct answer is (C).",
                            "Problem \n\tIn terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:\nOptions \n\t\n(A)::individualism and power distance. -- (B)::individualism. -- (C)::power distance and masculinity. -- (D)::uncertainty avoidance.\n Answer\n\tHofstede's model of cultural dimensions helps in understanding how values in the workplace are influenced by culture. This dimension reflects the degree to which a society emphasizes individual achievements and autonomy. In the U.S., there's a strong focus on personal freedom and self-reliance, distinguishing it from other dimensions like power distance, masculinity, or uncertainty avoidance.  Among the dimensions Hofstede identified, the United States stands out particularly for its high score in individualism. So the correct answer is (B).",
                            "Problem \n\tFor a stationary autoregressive process, shocks will\nOptions \n\t\n(A)::Eventually die away -- (B)::Persist indefinitely -- (C)::Grow exponentially -- (D)::Never occur\n Answer\n\tA stationary autoregressive process has a “memory” of past values, but the influence of any specific shock becomes smaller as time passes. Thus, shocks or temporary changes do not have a permanent effect on the series and they eventually die away over time. So the correct answer is (A).",
                            "Problem \n\tWhich of the following statements is NOT accurate regarding the services provided by local governments in the United States?\nOptions \n\t\n(A)::Duplication of efforts occurs often. -- (B)::Social problems of the central city spill over into the surrounding residential suburbs. -- (C)::Inefficiency in providing services occurs often. -- (D)::One neighborhood's efforts to reduce pollution are always supported by neighboring communities.\n Answer\n\tThere are several explanations, such as economic or political reasons, that neighboring communities would reject anti-pollution efforts. Thus, the statement that one neighborhood's efforts to reduce pollution are always supported by neighboring communities is not accurate in the context of local government services in the United States. So the correct answer is (D).",
                        ],
                        "Other": [
                            "Problem \n\tIn contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .\nOptions \n\t\n(A)::Buycotts, Boycotts, Blockchain technology, Charitable donations -- (B)::Buycotts, Boycotts, Digital technology, Increased Sales -- (C)::Boycotts, Buyalls, Blockchain technology, Charitable donations -- (D)::Boycotts, Buycotts, Digital technology, Increased Sales\n Answer\n\tBoycotts are campaigns that aim to punish or avoid companies for certain behaviors, whereas buycotts are campaigns that aim to reward favorable behavior by companies. The success of such buycott campaigns has been enhanced through digital technology, which facilitates increased sales for the company being supported. So the correct answer is (D).",
                            "Problem \n\tIn the assessment of the hand function which of the following is true?\nOptions \n\t\n(A)::Abduction of the thumb is supplied by spinal root T2 -- (B)::Opposition of the thumb by opponens policis is supplied by spinal root T1 -- (C)::Finger adduction is supplied by the median nerve -- (D)::Finger abduction is mediated by the palmar interossei\n Answer\n\tOpposition of the thumb is performed by the opponens pollicis muscle, which is supplied by the median nerve. The median nerve's spinal root is C8-T1. So the correct answer is (B). ",
                            "Problem \n\tWhat characteristic is not a key feature of the 'open systems' model of management?\nOptions \n\t\n(A)::Morale -- (B)::Innovation -- (C)::Growth resource -- (D)::Adaptation\n Answer\n\tThe 'open systems' model of management focuses on factors like innovation, growth resource, and adaptation, which are integral to an organization's interaction with its environment. Morale, while important in organizational management, is not a key feature of the open systems model specifically. So the correct answer is (A).",
                            "Problem \n\tWhen older adults move to a new state after retirement, which of the following is the more likely destination?\nOptions \n\t\n(A)::Texas -- (B)::California -- (C)::Hawaii -- (D)::Vermont\n Answer\n\tWhen older adults move to a new state after retirement, they often choose destinations with favorable climates and tax benefits. Texas, known for its warm climate and no state income tax, is a popular choice for retirees. So the correct answer is (A).",
                            "Problem \n\tWhich of these songs was a Top 10 hit for the rock band The Police?\nOptions \n\t\n(A)::'Radio Ga-Ga' -- (B)::'Ob-la-di Ob-la-da' -- (C)::'De Do Do Do De Da Da Da' -- (D)::'In-a-Gadda-Da-Vida'\n Answer\n\tA top 10 hit for the rock band The Police is 'De Do Do Do De Da Da Da'. This song is well-known and characteristic of their style. So the correct answer is (C).",
                        ],
                    },
                },
            }
            # print("Format number: ", self.format_num)
            # print("Category: ", mmlu_split[category])
            # print("CoT: ", self.CoT)
            shots = all_shots["CoT" if self.CoT else "vanilla"][str(self.format_num)][
                mmlu_split[category]
            ]  # get the specific few-shot prompt set for the MMLU category group of the question

        if shots:  # if this is a few-shot prompt
            return """{base}\n{shot1}\n{shot2}\n{shot3}\n{shot4}\n{shot5}\n""".format(
                base=self.base,
                shot1=shots[0],
                shot2=shots[1],
                shot3=shots[2],
                shot4=shots[3],
                shot5=shots[4],
            )
        elif self.CoT:  # if this is a Chain-of-Thought prompt
            return "Solve the problem and return (A), (B), (C) or (D)."
        elif self.base:  # if this is just a baseline prompt
            return "{base}".format(base=self.base)

    def __str__(self) -> str:
        return self.prompt

    def __repr__(self) -> str:
        return self.prompt

    def __hash__(self) -> int:
        return hash(self.prompt)

    def __eq__(self, other) -> bool:
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