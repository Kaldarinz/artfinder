import logging
import artfinder

logging.basicConfig(level=logging.DEBUG)

from artfinder.article import load_csv

cr = artfinder.Crossref()
af = artfinder.ArtFinder("aapopov1@mephi.ru")
df = load_csv('database/processed/kabashin_full.csv')

refs = []
for dois in df.loc[:3,'references']:
    if dois:
        refs.extend(dois)
refs = list(set(refs))
len(refs)

cr.get_dois(refs)
