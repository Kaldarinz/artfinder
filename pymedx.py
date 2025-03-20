import pymedx_custom
import logging

from pymedx_custom import PubMed
from pymedx_custom.helpers import pretty_print_xml, getContent
import logging
#logging.basicConfig(level=logging.DEBUG,datefmt='%H:%M:%S' , format='%(asctime)s.%(msecs)03d - %(module)s/%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')

pubmed = PubMed()

citin_articles = pubmed.getCitingArticles('28444919')

article = citin_articles[0]
print(article.abstract)