from artfinder import Crossref, load_csv, strict_filter
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
crosref = Crossref(app='artfinder', email='aapopov1@mephi.ru')

df = load_csv('database/processed/barcikowski_full.csv')
results, failed = crosref.get_refs(df[:5])