"""Module for handling articles."""

from __future__ import annotations

import datetime
import json

from typing import Any, Dict, List, Optional

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
        self.authors = self._extractAuthors(base)
        self.ISSN = getAllContent(element=self.xml, path=base + 'Journal/ISSN')
        self.journal = getAllContent(element=self.xml, path=base + 'Journal/Title')
        self.volume = getAllContent(element=self.xml, path=base + 'Journal/JournalIssue/Volume')
        self.issue = getAllContent(element=self.xml, path=base + 'Journal/JournalIssue/Issue')
        self.start_page = getAllContent(element=self.xml, path=base + 'Pagination/StartPage')
        self.end_page = getAllContent(element=self.xml, path=base + 'Pagination/EndPage')
        self.title = getAllContent(element=self.xml, path=base + 'ArticleTitle')
        
        # Pubmed data
        base = 'PubmedData/'
        self.doi = getAllContent(element=self.xml, path=base + 'ArticleIdList/ArticleId[@IdType="doi"]')
        self.pmcid = getAllContent(element=self.xml, path=base + 'ArticleIdList/ArticleId[@IdType="pmc"]')
        self.pmid = getAllContent(element=self.xml, path=base + 'ArticleIdList/ArticleId[@IdType="pubmed"]')
        self.publication_date = self._extractPublicationDate(base)

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

    def _extractKeywords(self) -> List[Any]:
        base = 'MedlineCitation/KeywordList/'
        return [
            keyword.text
            for keyword in self.xml.findall(base)
            if keyword is not None
        ]

    def _extractPublicationDate(self, base: str) -> Optional[datetime.date]:
        publication_date = self.xml.find(
            base + "History/PubMedPubDate[@PubStatus='accepted']"
        )
        if publication_date is not None:
            publication_year = getContent(publication_date, "./Year")
            publication_month = getContent(publication_date, "./Month")
            publication_day = getContent(publication_date, "./Day")

            if None in [publication_year, publication_month, publication_day]:
                return None
            date_str: str = (
                f"{publication_year}/{publication_month}/{publication_day}"
            )

            return datetime.datetime.strptime(date_str, "%Y/%m/%d")

        # Unable to parse the datetime
        return None

    def _extractReferences(self) -> List[dict[str, str]]:

        references = []
        for reference in self.xml.findall("PubmedData/ReferenceList/Reference"):
            references.append(
                {
                    'doi': getContent(reference, './/ArticleId[@IdType="doi"]', ''),
                    'pmid': getContent(reference, './/ArticleId[@IdType="pubmed"]', ''),
                    'pmcid': getContent(reference, './/ArticleId[@IdType="pmc"]', '')
                }   
            )
        return references
    
    def toDict(self) -> Dict[Any, Any]:
        """Convert the parsed information to a Python dict."""
        return {key: self.__getattribute__(key) for key in self.__slots__}

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


@typechecked
class PubMedCentralArticle:
    """Data class that contains a PubMedCentral article."""

    # Full slots
    """
    __slots__ = (
        "pmc_id",
        "title",
        "abstract",
        "keywords",
        "journal",
        "publication_date",
        "authors",
        "methods",
        "conclusions",
        "results",
        "copyrights",
        "doi",
        "xml",
    )
    """

    # slots which have been implemented
    __slots__ = (
        "abstract",
        "authors",
        "doi",
        "pmc_id",
        "publication_date",
        "title",
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

    def _extractPMCId(self, xml_element: _Element) -> Optional[str]:
        path = ".//article-meta/article-id[@pub-id-type='pmc']"
        return getContentUnique(element=xml_element, path=path)

    def _extractTitle(self, xml_element: _Element) -> Optional[str]:
        path = ".//title-group"
        return getAllContent(element=xml_element, path=path)

    # TODO: adapt the function for PubMed Central
    # def _extractKeywords(self, xml_element: _Element) -> List[Any]:
    #     path = ".//Keyword"
    #     return [
    #         keyword.text
    #         for keyword in xml_element.findall(path)
    #         if keyword is not None
    #     ]
    # TODO: adapt the function for PubMed Central
    # def _extractJournal
    # (self, xml_element: _Element) -> Optional[str]:
    #     path = ".//Journal/Title"
    #     return getContent(element=xml_element, path=path)

    def _extractAbstract(self, xml_element: _Element) -> Optional[str]:
        path = ".//abstract"
        return getAllContent(element=xml_element, path=path)

    # TODO: adapt the function for PubMed Central
    # def _extractConclusions(
    #     self, xml_element: _Element
    # ) -> Optional[str]:
    #     path = ".//AbstractText[@Label='CONCLUSION']"
    #     return getContent(element=xml_element, path=path)
    # TODO: adapt the function for PubMed Central
    # def _extractMethods(self, xml_element: _Element)
    # -> Optional[str]:
    #     path = ".//AbstractText[@Label='METHOD']"
    #     return getContent(element=xml_element, path=path)
    # TODO: adapt the function for PubMed Central
    # def _extractResults(self, xml_element: _Element)
    # -> Optional[str]:
    #     path = ".//AbstractText[@Label='RESULTS']"
    #     return getContent(element=xml_element, path=path)
    # TODO: adapt the function for PubMed Central
    # def _extractCopyrights(
    #     self, xml_element: _Element
    # ) -> Optional[str]:
    #     path = ".//CopyrightInformation"
    #     return getContent(element=xml_element, path=path)

    def _extractDoi(self, xml_element: _Element) -> Optional[str]:
        path = ".//article-meta/article-id[@pub-id-type='doi']"
        return getContentUnique(element=xml_element, path=path)

    def _extractPublicationDate(
        self, xml_element: _Element
    ) -> Optional[datetime.date]:
        # Get the publication date

        # Get the publication elements
        publication_date = xml_element.find(".//pub-date[@pub-type='epub']")

        if publication_date is None:
            publication_date = xml_element.find(".//pub-date")

        if publication_date is not None:
            publication_year = getContent(publication_date, ".//year", None)

            if not publication_year or publication_year is None:
                return None

            publication_month = getContent(publication_date, ".//month", "1")

            publication_day = getContent(publication_date, ".//day", "1")

            # Construct a datetime object from the info
            date_str: str = (
                f"{str(publication_year).strip()}/"
                f"{str(publication_month).strip()}/"
                f"{str(publication_day).strip()}"
            )

            return datetime.datetime.strptime(date_str, "%Y/%m/%d")

        # Unable to parse the datetime
        return None

    def _extractAuthors(
        self, xml_element: _Element
    ) -> List[dict[str, Optional[str]]]:
        contrib_group = xml_element.findall(".//contrib-group")
        if contrib_group:
            return [
                {
                    "lastname": getContent(author, ".//surname", None),
                    "firstname": getContent(author, ".//given-names", None),
                    # TODO: adapt the function for PubMed Central
                    # "initials": getContent(author, ".//Initials", None),
                    # "affiliation": getContent(
                    #     author, ".//AffiliationInfo/Affiliation", None
                    # ),
                }
                for author in contrib_group[0].findall(
                    ".//contrib[@contrib-type='author']"
                )
            ]
        return []

    def _initializeFromXML(self, xml_element: _Element) -> None:
        """Parse an XML element into an article object."""
        # Parse the different fields of the article
        self.pmc_id = self._extractPMCId(xml_element)
        self.title = self._extractTitle(xml_element)
        self.abstract = self._extractAbstract(xml_element)
        self.doi = self._extractDoi(xml_element)
        self.publication_date = self._extractPublicationDate(xml_element)
        self.authors = self._extractAuthors(xml_element)
        # TODO: adapt the function for PubMed Central
        # self.xml = xml_element
        # self.keywords = self._extractKeywords(xml_element)
        # self.journal = self._extractJournal(xml_element)
        # self.conclusions = self._extractConclusions(xml_element)
        # self.methods = self._extractMethods(xml_element)
        # self.results = self._extractResults(xml_element)
        # self.copyrights = self._extractCopyrights(xml_element)

    def toDict(self) -> Dict[Any, Any]:
        """Convert the parsed information to a Python dict."""
        return {key: self.__getattribute__(key) for key in self.__slots__}

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
