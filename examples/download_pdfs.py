'''from xml.etree import ElementTree as ET

def sanitize(inputString):
    cleaned_text = ''.join(char for char in inputString if char.isascii())
    return cleaned_text

f = open("arxiv_gpt 3_data.xml", "rb")
f = f.read().decode("utf-8","ignore")

print(type(f))
#f = sanitize(f)
tree = ET.fromstring(f)
print(tree)'''

import json
from prompt_systematic_review.paperSource import Paper
from prompt_systematic_review import arxiv_source
import tqdm
import requests
#read json file called arxiv.json
#mydata = json.load(open("arxiv.json"))
from multiprocessing import Pool
import pandas as pd
import time

#read json file called arxiv.json to dataframe
df = pd.read_csv("arxiv_clean.csv")
print(df.head())

"""def downloadPaper(paper):
    url = paper.url + ".pdf"
    print(url)
    response = requests.get(url)
    print(response)
    paperName = paper.url.split("/")[-1] + ".pdf"
    print(paperName)
    with open("./arxivPDFs/" + paperName, 'wb') as f:
        f.write(response.content)
"""

if __name__ == '__main__':
    aSource = arxiv_source.ArXivSource()
    with Pool(processes=16) as pool:
        for index, row in tqdm.tqdm(df.iterrows()):
            paper = Paper(row['title'], row['firstAuthor'], row['url'], row['dateSubmitted'], row['keywords'])
            pool.apply_async(aSource.getPaperSrc, args = (paper, "./arxivPDFs/"))
          
            time.sleep(0.5)



    #download pdf from paper.url using requests module
    #save pdf to local folder
   
    #left off at 6427





