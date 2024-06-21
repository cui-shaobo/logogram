import requests
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
import jsonlines

from collection import fetch_webpage, parse_titles, generate_datasets

# emnlp
def generate_all_emnlp(year):
    root_url = f"https://aclanthology.org/events/emnlp-{year}/"
    root_page = requests.get(root_url).text
    root_page_soup = BeautifulSoup(root_page, "html.parser")
    paper_lists = root_page_soup.find_all('strong')  # all papers in this year

    a_lists = []
    for paper in paper_lists:
        res = paper.find('a', {'class':'align-middle'})
        if res != None:
            a_lists.append(res)

    url_lists = [ a.get('href') for a in a_lists]

    results = []
    title_pattern = re.compile(r'title = &#34;(.*?)&#34')
    abs_pattern = re.compile(r'Abstract</h5><span>(.*?)</span></div>')

    for url in url_lists:
        # print(f'https://aclanthology.org{url}')
        response = requests.get(f'https://aclanthology.org{url}')
        data = response.text
        titles = title_pattern.findall(data)[0].replace('{', '').replace('}', '')
        if ':' in titles:
            abbr, title = parse_titles(titles)
            abstracts = abs_pattern.findall(data)
            if len(abstracts) == 0:
                continue
            else:
                abstract = abstracts[0]
            if abbr != None and title != None and abstract != None:
                new_item = {'Type': 'conference', 'Year': f'{year}', 'Area': 'AI', 'Where': 'EMNLP', 'Abbreviation': abbr,
                        'Title': title, 'Abstract': abstract}
                results.append(new_item)

    generate_datasets(results, file_path=f'./results/data_emnlp_{year}.jsonl')

