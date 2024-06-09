import requests
from xml.etree import ElementTree as ET
import re

# custom header so requests don't get blocked
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0"
}


def process_paper_title(title: str) -> str:
    """
    Process a paper title by converting it to lowercase, removing newline characters,
    replacing '-' with '', collapsing any amount of spaces in a row to a single space.

    :param title: The original title of the paper.
    :type title: str
    :return: The processed title.
    :rtype: str
    """
    # Replace '-' with '', collapse multiple spaces to a single space, and strip leading/trailing spaces
    s = (
        re.sub(r"\s+", " ", title.lower().replace("\n", "").replace("-", ""))
        .strip()
        .replace(".", "")
        .replace("/", "")
        .replace("\\", "")
        .replace(":", "")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
    )
    # .replace(" ", "")
    # s = re.sub(r'\W+', '', s)
    return s

