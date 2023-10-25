import requests
from xml.etree import ElementTree as ET

def search_arxiv(keyword):
    url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}&start=0&max_results=10000"
    data = requests.get(url).content
    return data

def count_articles(data):
    root = ET.fromstring(data)
    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
    return len(entries)


