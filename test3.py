import logging
import artfinder

#logging.basicConfig(level=logging.DEBUG)

from artfinder.article import load_csv

cr = artfinder.Crossref()
af = artfinder.ArtFinder()

df = af.search('TiN nanoparticles', author='kabashin', pub_since='2018', max_results=None)
new_obj = af.get_refs(df[:2])
