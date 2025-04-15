import logging
import artfinder

logging.basicConfig(level=logging.DEBUG)

cr = artfinder.Crossref()
af = artfinder.ArtFinder("aapopov1@mephi.ru")
article_by_doi = af.find_article(doi='10.1364/ol.404304')
