import logging
import artfinder

logging.basicConfig(level=logging.DEBUG)

from artfinder.article import load_csv

cr = artfinder.Crossref()
af = artfinder.ArtFinder()

af.search('laser synthesis of colloids', pub_since='2010')
