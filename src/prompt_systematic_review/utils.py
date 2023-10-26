import requests
from xml.etree import ElementTree as ET

def search_arxiv(keyword,max_results=10000):
    url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}&start=0&max_results={max_results}"
    data = requests.get(url).content
    return data

def count_articles(data):
    root = ET.fromstring(data)
    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
    return len(entries)


