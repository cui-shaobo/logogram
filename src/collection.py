import requests
from bs4 import BeautifulSoup
import jsonlines
import re

def fetch_webpage(url):
    response = requests.get(url)
    return response.text

def parse_titles(titles):
    colon_index = titles.index(':')
    abbreviation = titles[:colon_index]
    title = titles[colon_index+1:].lstrip()
    return abbreviation, title

def generate_datasets(results, file_path):
    with jsonlines.open(file_path, 'w') as f:
        for sample in results:
            f.write(sample)
# nips
def fetch_content_from_nips(text, year):
    soup = BeautifulSoup(text, "html.parser")
    ul_list = soup.find('ul', {'class':'paper-list'})
    a_list = ul_list.find_all('a')
    results = []
    for content in a_list:
        titles = content.text.strip()
        if ':' in titles:
            href = content.get('href')
            abstract = fetch_abstract_from_nips('https://papers.nips.cc' + href)
            abbr, title = parse_titles(titles)
            new_item = {'Type':'conference', 'Year':f'{year}', 'Area':'AI', 'Where':'NeurIPS', 'Abbreviation':abbr,
                        'Title': title, 'Abstract':abstract}
            results.append(new_item)
    return results

def fetch_abstract_from_nips(url):
    text = fetch_webpage(url)
    soup = BeautifulSoup(text, "html.parser")
    div_list = soup.find('div', {'class': 'container-fluid'})
    p_list = div_list.find_all('p')
    abstract = p_list[-1].text.strip()
    return abstract

def generate_nips_datsets(year):
    url = f'https://papers.nips.cc/paper_files/paper/{year}'
    web_content = fetch_webpage(url)
    results = fetch_content_from_nips(web_content, year)
    print(len(results))
    with jsonlines.open(f'./results/data_nips_{year}.jsonl', 'w') as f:
        for sample in results:
            f.write(sample)

# acl
def generate_all_acl(year):
    root_url = f"https://aclanthology.org/events/acl-{year}/"
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
                new_item = {'Type': 'conference', 'Year': f'{year}', 'Area': 'AI', 'Where': 'ACL', 'Abbreviation': abbr,
                        'Title': title, 'Abstract': abstract}
                results.append(new_item)

    generate_datasets(results, file_path=f'./results/data_acl_{year}.jsonl')



if __name__ == "__main__":
    for i in range(1979, 2022):
        generate_all_acl(i)
