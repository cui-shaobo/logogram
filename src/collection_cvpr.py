from tqdm import tqdm
from bs4 import BeautifulSoup
import jsonlines

from collection import fetch_webpage, parse_titles

# cvpr
def fetch_content_from_cvpr(text, year):
    soup = BeautifulSoup(text, "html.parser")
    dt_list = soup.find_all('dt', {'class': 'ptitle'})
    print('\n\n')
    print(f'year: {year}')
    print(f'num_papers: {len(dt_list)}')
    results = []
    for content in tqdm(dt_list):
        titles = content.text.strip()
        if ':' in titles:
            href = content.a['href']
            abstract = fetch_abstract_from_cvpr('https://openaccess.thecvf.com/' + href)
            if abstract:
                abbr, title = parse_titles(titles)
                new_item = {'Type':'conference', 'Year':f'{year}', 'Area':'CV', 'Where':'CVPR', 'Abbreviation':abbr,
                            'Title': title, 'Abstract':abstract}
                results.append(new_item)
    return results

def fetch_abstract_from_cvpr(url):
    text = fetch_webpage(url)
    soup = BeautifulSoup(text, "html.parser")
    div_tag = soup.find('div', {'id': 'abstract'})
    if div_tag is not None:
        abstract = div_tag.text.strip()
        return abstract
    else:
        return None

def generate_cvpr_datsets(year):
    url = f'https://openaccess.thecvf.com/CVPR{year}'
    web_content = fetch_webpage(url)
    results = fetch_content_from_cvpr(web_content, year)
    print(len(results))
    with jsonlines.open(f'./results/data_cvpr_{year}.jsonl', 'w') as f:
        for sample in results:
           f.write(sample)

def extract_date_urls(web_content):
    soup = BeautifulSoup(web_content, 'html.parser')
    content_div = soup.find('div', {'id': 'content'})

    date_hrefs = []
    for link in content_div.find_all('a'):
        href = link.get('href')
        date_hrefs.append(href)

    return date_hrefs


def generate_cvpr_dataset_after_2017(year):
    assert year > 2017
    url = f'https://openaccess.thecvf.com/CVPR{year}'
    short_url = 'https://openaccess.thecvf.com/'
    web_content = fetch_webpage(url)
    day_hrefs = extract_date_urls(web_content)
    full_hrefs = [short_url + href for href in day_hrefs]
    print('full_hrefs: {}'.format(full_hrefs))
    for i, href in enumerate(full_hrefs):
        web_content = fetch_webpage(href)
        results = fetch_content_from_cvpr(web_content, year)
        print(len(results))
        with jsonlines.open(f'./results/data_cvpr_{year}_{i}.jsonl', 'w') as f:
            for sample in results:
               f.write(sample)

def generate_cvpr_dataset_after_2020(year):
    assert year > 2020
    url = f'https://openaccess.thecvf.com/CVPR{year}?day=all'
    web_content = fetch_webpage(url)
    results = fetch_content_from_cvpr(web_content, year)
    print(len(results))
    with jsonlines.open(f'./results/data_cvpr_{year}.jsonl', 'w') as f:
        for sample in results:
           f.write(sample)

    with jsonlines.open(f'./results/data_cvpr_{year}.jsonl', 'w') as f:
        for sample in results:
           f.write(sample)

if __name__ == "__main__":
    # From year 2013 to year 2017, these papers in CVPR.
    # for year in range(2013, 2018):
    #     generate_cvpr_datsets(year)

    # From year 2018 to year 2020, these papers are arranged by day order.
    # for year in range(2018, 2021):
    #     generate_cvpr_dataset_after_2017(year)

    # From year 2021 to year 2020, these papers are arranged by day order.
    for year in range(2022, 2023):
        generate_cvpr_dataset_after_2020(year)


