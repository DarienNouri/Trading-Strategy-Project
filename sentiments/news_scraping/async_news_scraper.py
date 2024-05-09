import asyncio
import csv
import os
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import json
from urllib.parse import urlencode
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


"""
Get cookies from 'https://persistent.library.nyu.edu/arch/NYU02479'

"""

# Define the base URL and other constants
base_url = "http://advance.lexis.com/api/search"

cookies = {

	".AspNet.Wam": "20b3ca4a-73ee-4e90-a6e8-e8971f4c0170:nu",
		"ASP.NET_SessionId": "f3ef98cb-c22f-4318-937c-e185d9991c09",
	"LexisMachineId": "cd02aa08-5128-400b-a97d-f5daa06f9dcd",
	"lna2": "OTBmODkwYTQ5NTYxZDFkZTgyZDk1M2Q2YjhkNzczZTA1ZTA4MzU1OTZhNGIwOThlODI2NTRiNDdjNTVlM2M1ODY2MGE0ZjkydXJuOnVzZXI6UEExODc2NzA5ODMhMTAwMDIwMiwxNTE2ODIzLDE1MTY4MzEsIW5vbmVe",
	"LNPAGEHISTORY": "ddc50e3a-5073-45d7-83f6-2bf0ad83585a,4b5b5069-6853-4d45-989d-c683f633de38,9a9bcb7c-db8f-4f6e-b3c3-a5d896ebd443,665b2c57-7458-41d6-ad85-7c8bc4915a8a,62571327-fb59-40ea-8e92-73f610ffda05,e655b9d2-51f5-4152-af85-ec4ca1304dcf",
	"X-LN-Session-TTL": "2024-04-01T09:51:06Z,2024-04-01T06:51:06Z"
	}

# The keys we want to keep from the cookies
keep_cookie_keys = ['.AspNet.Wam', 'ASP.NET_SessionId', 'LexisMachineId', 'lna2', 'LNPAGEHISTORY', 'X-LN-Session-TTL']

# Raw cookies should look like this. make sure you are on NYU network/VPN
raw_cookies = 'LexisMachineId=cd02aa08-5128-400b-a97d-f5daa06f9dcd; X-LN-Session-TTL=2024-04-01T23%3A24%3A14Z%2C2024-04-01T20%3A24%3A14Z; LNPAGEHISTORY=e655b9d2-51f5-4152-af85-ec4ca1304dcf%2Ce5a4bd82-67d8-4e3d-8741-46ab922d7520%2C87778d31-2c6f-4f0a-b4b1-76c42690cca5%2C87dfdc32-1f54-4f6f-a071-a4fe1630d3d1%2Cb918b1c6-0611-426e-83c3-6559a9a73b54%2C8336b83f-df0a-4288-853e-cb9fa656b71c; bisNexisLocalStore=%7B%22data%22%3A%5B%7B%22title%22%3A%22Nexis%20Uni%C2%AE%20Home%22%2C%22url%22%3A%22%2Fbisacademicresearchhome%3Fcrid%3D8336b83f-df0a-4288-853e-cb9fa656b71c%26pdmfid%3D1516831%26pdisurlapi%3Dtrue%22%7D%2C%7B%22title%22%3A%22Results%3A%20(tesla)%22%2C%22url%22%3A%22%22%7D%5D%7D; ASP.NET_SessionId=f3ef98cb-c22f-4318-937c-e185d9991c09; lna2=MTc1ODljMjJhNGE2NDEzODA2ZjJlZTFlOTBkMTcxZjg5YjZhNjljMjE0MDZlNDkxZmU2ZjY2OTE3MWMyOGIzNjY2MGFmYmNmdXJuOnVzZXI6UEExODc2NzA5NzkhMTAwMDIwMiwxNTE2ODIzLDE1MTY4MzEsIW5vbmVe; .AspNet.Wam=4a7a6a30-61d0-4b8f-9a1b-ff1152b02c32%3Anu; Perf=%7B%22name%22%3A%22gns_search_box-search_run%22%2C%22sit%22%3A%221711995853149.867%22%7D; originalsearchdata=%7B%22Product%22%3A%22Nexis%20Uni%22%2C%22Page%22%3A%22bisacademicresearchhome%22%2C%22SearchTerms%22%3A%22tesla%22%2C%22AppliedFilters%22%3A%5B%7B%22displayText%22%3A%22All%20available%20dates%22%2C%22name%22%3A%22datestr%22%2C%22value%22%3A%22alldates%22%7D%5D%7D'


def parse_cookies(raw_cookies):
    cookies_dict = {}
    for pair in raw_cookies.split('; '):
        key, value = pair.split('=')
        if key in keep_cookie_keys:
            cookies_dict[key] = value
    return cookies_dict

cookies = parse_cookies(raw_cookies)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_to_csv(filename, data_rows):
    # Check if file exists to write headers or not
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(data_rows[0].keys()) if data_rows else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in data_rows:
            writer.writerow(row)


async def fetch_data(session, query_params, cookies):
    search_params_url = urlencode(query_params)
    full_url = f"{base_url}?{search_params_url}"
    async with session.get(full_url, cookies=cookies) as response:
        return await response.text()

def parse_response(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_script = soup.find('script', text=lambda text: text and 'this.add(\'page.model\'' in text)
    if not target_script:
        return []
    script_content = target_script.string
    start_index = script_content.find("{\"id\":\"pagemodel\"")
    end_index = script_content.rfind('}}}') + 3
    json_str = script_content[start_index:end_index]
    content = json.loads(json_str)
    article_rows = content['collections']['componentmodels']['collections']['featureproviders'][0]['collections']['results']['collections']['rows']
    actualresultscount = content['collections']['componentmodels']['collections']['featureproviders'][0]['props']['actualresultscount']
    [article['props'].update({'actualresultscount': actualresultscount}) for article in article_rows]

    return article_rows

def extract_row_data(row):
    keys_to_extract = ["title", "docid", "updated", "publisheddate", "relevancescore", "resultnumber", "jurisdiction", "date", "source", "author", "contenttype", "status", "contentcomponentid", "vendorreportid", "length", "byline", "wordcount", "language", "actualresultscount"]
    return {key: row['props'].get(key, None) for key in keys_to_extract}



async def main(search_terms, start_date, end_date, cookies):
    articles = []  # List to hold all articles for DataFrame
    async with aiohttp.ClientSession() as session:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        delta = timedelta(days=1)
        while start_dt <= end_dt:
            query_params = {
                'q': search_terms,
                'collection': collection,
                'jurisdiction': 'dXJuOmpwZjoxMDk5',
                'startdate': start_dt.strftime("%Y-%m-%d"),
                'enddate': start_dt.strftime("%Y-%m-%d"),
            }
            html_content = await fetch_data(session, query_params, cookies)
            article_rows = parse_response(html_content)
            extracted_data = [extract_row_data(row) for row in article_rows]
            articles.extend(extracted_data)  # Append to articles list for DataFrame
            if extracted_data:
                filename = f'./results/{search_terms}_{start_date}_{end_date}.csv'
                ensure_dir(filename)
                write_to_csv(filename, extracted_data)
            start_dt += delta
    return pd.DataFrame(articles)




def date_range(start_date, end_date, interval):
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=interval)

async def main(search_terms, start_date, end_date, cookies, interval=7, save_dir='../results'):
    articles = []  # List to hold all articles for DataFrame
    async with aiohttp.ClientSession() as session:
        for start_dt in date_range(start_date, end_date, interval):
            end_dt = min(start_dt + timedelta(days=interval-1), datetime.strptime(end_date, "%Y-%m-%d"))
            query_params = {
                'q': search_terms,
                'collection': collection,
                'jurisdiction': 'dXJuOmpwZjoxMDk5',
                'startdate': start_dt.strftime("%Y-%m-%d"),
                'enddate': end_dt.strftime("%Y-%m-%d"),
            }
            html_content = await fetch_data(session, query_params, cookies)
            article_rows = parse_response(html_content)
            extracted_data = [extract_row_data(row) for row in article_rows]
            articles.extend(extracted_data)  # Append to articles list for DataFrame
            if extracted_data:
                filename = f'{save_dir}/{search_terms}_{start_date}_{end_date}.csv'
                ensure_dir(filename)
                write_to_csv(filename, extracted_data)
    return pd.DataFrame(articles)


# Adjust these dates as needed
portfolio_symbol_basket = [
        'GE',
        'HON',
        'DE',
        'UNP',
        'FDX',
        'JPM',
        'BAC',
        'WFC',
        'C',
        'GS',
        'MS',
        'AXP',
        'BLK',
        'AAPL',
        'GOOGL',
        'INTC',
        'CRM',
 ]

start_date = '2016-01-01'
end_date = '2024-03-31'
query_interval = 7 # days
search_terms = "GE"
search_terms = portfolio_symbol_basket[1]

collection = "news"

# Uncomment to run from here
"""
df = asyncio.run(main(search_terms, start_date, end_date, interval=query_interval))
print(df)
"""
