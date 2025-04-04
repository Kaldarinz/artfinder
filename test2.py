from artfinder import Crossref, load_csv, strict_filter
from artfinder.helpers import full_texts_from_urls
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
crosref = Crossref(app='artfinder', email='aapopov1@mephi.ru')

def get_pdf_link(link_entry: list[dict]|None) -> str|None:
    """
    Extracts the PDF link from the provided link entry."
    """

    if link_entry is None or len(link_entry) == 0:
        return None
    for entry in link_entry:
        if entry.get('content-type') == 'application/pdf':
            return entry.get('url')

df = load_csv('database/processed/barcikowski_full.csv')
df = df.sort_values(by='publication_date', ascending=False).reset_index(drop=True)
df['full_text_link'] = df['link'].map(get_pdf_link)
full_text_links = df['full_text_link'].dropna()
urls = full_text_links.iloc[:2].to_list()
full_texts_from_urls(urls)
