"""Module for handling articles."""

from __future__ import annotations

import datetime
import json

from typing import Any, Dict, List, Optional, cast

from lxml.etree import _Element
from typeguard import typechecked

from .helpers import getAbstract, getAllContent, getContent, getContentUnique


@typechecked
class PubMedArticle:
    """Data class that contains a PubMed article."""

    __slots__ = (
        "abstract",
        "authors",
        "ISSN",
        "journal",
        "volume",
        "issue",
        "start_page",
        "end_page",
        "doi",
        "pmid",
        "pmcid",
        "publication_date",
        "references",
        "title",
        "keywords",
        "type",
        "xml",
    )

    def __init__(
        self,
        xml_element: Optional[_Element] = None,
        *args: List[Any],
        **kwargs: Dict[Any, Any],
    ) -> None:
        """Initialize of the object from XML or from parameters."""
        if args:
            # keep it for resolving problems with linter
            pass
        # If an XML element is provided, use it for initialization
        if xml_element is not None:
            self._initializeFromXML(xml_element=xml_element)

        # If no XML element was provided, try to parse the input parameters
        else:
            for field in self.__slots__:
                self.__setattr__(field, kwargs.get(field, None))

    def _initializeFromXML(self, xml_element: _Element) -> None:
        """Parse an XML element into an article object."""
        # Parse the different fields of the article
        self.xml = xml_element
        # MedlineCitation data
        base = 'MedlineCitation/Article/'
        self.abstract = getAllContent(element=self.xml, path=base + 'Abstract/AbstractText')
        self.ISSN = getAllContent(element=self.xml, path=base + 'Journal/ISSN')
        self.journal = getAllContent(element=self.xml, path=base + 'Journal/Title')
        self.volume = getAllContent(element=self.xml, path=base + 'Journal/JournalIssue/Volume')
        self.issue = getAllContent(element=self.xml, path=base + 'Journal/JournalIssue/Issue')
        self.start_page = getAllContent(element=self.xml, path=base + 'Pagination/StartPage')
        self.end_page = getAllContent(element=self.xml, path=base + 'Pagination/EndPage')
        self.title = getAllContent(element=self.xml, path=base + 'ArticleTitle')
        self.type = getAllContent(element=self.xml, path=base + 'PublicationTypeList/PublicationType')
        self.authors = self._extractAuthors(base)
        self.publication_date = self._extractPublicationDate(base)
        
        # Pubmed data
        base = 'PubmedData/'
        self.doi = getAllContent(element=self.xml, path=base + 'ArticleIdList/ArticleId[@IdType="doi"]')
        self.pmcid = getAllContent(element=self.xml, path=base + 'ArticleIdList/ArticleId[@IdType="pmc"]')
        self.pmid = getAllContent(element=self.xml, path=base + 'ArticleIdList/ArticleId[@IdType="pubmed"]')

        # Other data
        self.keywords = self._extractKeywords()
        self.references = self._extractReferences()

    def _extractAuthors(self, base_path: str) -> List[dict[str, Optional[str]]]:
        base_path += 'AuthorList/'
        return [
            {
                "lastname": getContent(author, "./LastName"),
                "firstname": getContent(author, "./ForeName"),
                "initials": getContent(author, "./Initials"),
                "affiliation": getContent(author, "./AffiliationInfo/Affiliation"),
            }
            for author in self.xml.findall(base_path + "Author")
        ]

    def _extractKeywords(self) -> List[str]|None:
        base = 'MedlineCitation/KeywordList/'
        result = [
            keyword.text
            for keyword in self.xml.findall(base)
            if (keyword is not None and keyword.text is not None)
        ]
        return result if result else None

    def _extractPublicationDate(self, base: str) -> Optional[datetime.date]:
        publication_date = self.xml.find(
            base + "ArticleDate"
        )
        # First try to get the publication date from the ArticleDate tag
        if publication_date is not None:
            publication_year = getContent(publication_date, "./Year")
            publication_month = str(getContent(publication_date, "./Month", "01"))
            publication_day = str(getContent(publication_date, "./Day", "01"))

            if None in [publication_year, publication_month, publication_day]:
                return None
            date_str: str = (
                f"{publication_year}/{publication_month}/{publication_day}"
            )

            return datetime.datetime.strptime(date_str, "%Y/%m/%d")

        # If the ArticleDate tag is not present, try to get the publication date from the PubDate tag
        elif (publication_Date:=self.xml.find(base + "Journal/JournalIssue/PubDate")) is not None:
            # Try to get the publication date from the PubDate tag
            publication_year = getContent(publication_Date, "./Year")
            publication_month = getContent(publication_Date, "./Month", "Jan")
            publication_day = getContent(publication_Date, "./Day", "01")

            # Date can have alternative MedlineDate format
            if publication_year is None:
                date = getContentUnique(publication_Date, "./MedlineDate")
                if date is not None:
                    date = date.split(" ")
                    publication_year = date[0]
                    publication_month = date[1] if len(date) > 1 else "Jan"
                    if len(date) > 2:
                        publication_day = date[2].split('-')[0]
                    else:
                        publication_day = "01"
                else:
                    return None

            date_str = '/'.join([publication_year, publication_month, publication_day])

            return datetime.datetime.strptime(date_str, "%Y/%b/%d")
        return None

    def _extractReferences(self) -> List[dict[str, str]]|None:

        references = []
        for reference in self.xml.findall("PubmedData/ReferenceList/Reference"):
            references.append(
                {
                    'doi': getContent(reference, './/ArticleId[@IdType="doi"]', ''),
                    'pmid': getContent(reference, './/ArticleId[@IdType="pubmed"]', ''),
                    'pmcid': getContent(reference, './/ArticleId[@IdType="pmc"]', '')
                }   
            )
        return references if references else None
    
    def toDict(self) -> Dict[Any, Any]:
        """Convert the parsed information to a Python dict."""
        return {key: self.__getattribute__(key) for key in self.__slots__ if key != 'xml'}

    def toJSON(self) -> str:
        """Dump the object as JSON string."""
        return json.dumps(
            {
                key: (
                    value
                    if not isinstance(value, (datetime.date, _Element))
                    else str(value)
                )
                for key, value in self.toDict().items()
            },
            sort_keys=True,
            indent=4,
        )
