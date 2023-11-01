import requests
from xml.etree import ElementTree as ET

# custom header so requests don't get blocked
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0'
}

def search_arxiv(keyword,max_results=10000):
    url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}&start=0&max_results={max_results}"
    data = requests.get(url).content
    return data

def count_articles(data):
    root = ET.fromstring(data)
    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
    return len(entries)



