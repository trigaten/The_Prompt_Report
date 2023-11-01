import requests
from xml.etree import ElementTree as ET

# custom header so requests don't get blocked
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0"
}


def process_paper_title(title: str) -> str:
    """
    Process a paper title by converting it to lowercase and removing newline characters and extra spaces.

    :param title: The original title of the paper.
    :type title: str
    :return: The processed title.
    :rtype: str
    """
    return title.lower().replace("\n", "").replace("  ", " ")
