"""Module for handling articles."""

from __future__ import annotations

import datetime
import json
import re
from ast import literal_eval

from typing import Any, Dict, List, Optional, cast, Iterable

from lxml.etree import _Element
from typeguard import typechecked
import pandas as pd
from pandas import DataFrame

from artfinder.helpers import getAllContent, getContent, getContentUnique


# TODO: There should probably be only one Article class
@typechecked
class Article:
    """Base class for all articles."""

    __slots__ = (
        "title",
        "authors",
        "journal",
        "publication_date",
        "link",
        "doi",
        "type",
        "keywords",
        "is_referenced_by_count",
        "abstract",
        "publisher",
        "issn",
        "volume",
        "issue",
        "start_page",
        "end_page",
        "references",
        "pmid",
        "pmcid",
        "license",
    )

    def __init__(self) -> None:
        """Initialize all attributes in __slots__ to None."""
        for slot in self.__slots__:
            setattr(self, slot, None)

    def to_dict(self) -> Dict[Any, Any]:
        """Convert the parsed information to a Python dict."""
        dct = {key: self.__getattribute__(key) for key in self.get_all_slots()}
        for key, val in dct.items():
            if val is not None:
                dct[key] = str(val).lower()
        return dct

    @classmethod
    def get_all_slots(cls):
        """
        Get all __slots__ of a class, including inherited ones.

        Parameters
        ----------
        cls : type
            The class to inspect.

        Returns
        -------
        list
            A list of all __slots__ defined in the class and its superclasses.
        """
        slots = []
        for base in cls.__mro__:  # Traverse the Method Resolution Order (MRO)
            if hasattr(base, "__slots__"):
                slots.extend(base.__slots__)
        return slots

    @staticmethod
    def col_types() -> dict[str, str]:
        """Return dictionary with column names and their types."""
        return {
            "abstract": "string",
            "title": "string",
            "doi": "string",
            "type": "string",
            "journal": "string",
            "issn": "string",
            "volume": "string",
            "issue": "string",
            "start_page": "string",
            "end_page": "string",
            "is_referenced_by_count": "int",
        }


@typechecked
class PubMedArticle(Article):
    """Data class that contains a PubMed article."""

    def __init__(self, xml_element: Optional[_Element] = None, **kwargs: Any) -> None:
        """Initialize of the object from XML or from parameters."""

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
        # MedlineCitation data
        base = "MedlineCitation/Article/"
        self.abstract = getAllContent(
            element=xml_element, path=base + "Abstract/AbstractText"
        )
        self.issn = getAllContent(element=xml_element, path=base + "Journal/ISSN")
        self.journal = getAllContent(element=xml_element, path=base + "Journal/Title")
        self.volume = getAllContent(
            element=xml_element, path=base + "Journal/JournalIssue/Volume"
        )
        self.issue = getAllContent(
            element=xml_element, path=base + "Journal/JournalIssue/Issue"
        )
        self.start_page = getAllContent(
            element=xml_element, path=base + "Pagination/StartPage"
        )
        self.end_page = getAllContent(
            element=xml_element, path=base + "Pagination/EndPage"
        )
        self.title = getAllContent(element=xml_element, path=base + "ArticleTitle")
        self.type = getAllContent(
            element=xml_element, path=base + "PublicationTypeList/PublicationType"
        )
        if self.type == "Journal Article":
            self.type = "article-journal"
        self.authors = self._extractAuthors(element=xml_element, base_path=base)
        self.publication_date = self._extractPublicationDate(
            element=xml_element, base=base
        )

        # Pubmed data
        base = "PubmedData/"
        self.doi = getAllContent(
            element=xml_element, path=base + 'ArticleIdList/ArticleId[@IdType="doi"]'
        )
        self.pmcid = getAllContent(
            element=xml_element, path=base + 'ArticleIdList/ArticleId[@IdType="pmc"]'
        )
        self.pmid = getAllContent(
            element=xml_element, path=base + 'ArticleIdList/ArticleId[@IdType="pubmed"]'
        )

        # Other data
        self.keywords = self._extractKeywords(xml_element)
        self.references = self._extractReferences(xml_element)

    def _extractAuthors(
        self, element: _Element, base_path: str
    ) -> List[dict[str, Optional[str]]]:
        base_path += "AuthorList/"
        return [
            {
                "lastname": getContent(author, "./LastName"),
                "firstname": getContent(author, "./ForeName", "")
                + getContent(author, "./Initials", ""),
                "affiliation": getContent(author, "./AffiliationInfo/Affiliation"),
            }
            for author in element.findall(base_path + "Author")
        ]

    def _extractKeywords(self, element: _Element) -> List[str] | None:
        base = "MedlineCitation/KeywordList/"
        result = [
            keyword.text
            for keyword in element.findall(base)
            if (keyword is not None and keyword.text is not None)
        ]
        return result if result else None

    def _extractPublicationDate(
        self, element: _Element, base: str
    ) -> Optional[datetime.date]:
        publication_date = element.find(base + "ArticleDate")
        # First try to get the publication date from the ArticleDate tag
        if publication_date is not None:
            year = getContent(publication_date, "./Year")
            month = str(getContent(publication_date, "./Month", "01"))
            day = str(getContent(publication_date, "./Day", "01"))

            if year is not None:
                return datetime.date(int(year), int(month), int(day))
        # If the ArticleDate tag is not present, try to get the publication date from the PubDate tag
        elif (
            publication_Date := element.find(base + "Journal/JournalIssue/PubDate")
        ) is not None:
            # Try to get the publication date from the PubDate tag

            month_abbr_to_num = {
                "Jan": 1,
                "Feb": 2,
                "Mar": 3,
                "Apr": 4,
                "May": 5,
                "Jun": 6,
                "Jul": 7,
                "Aug": 8,
                "Sep": 9,
                "Oct": 10,
                "Nov": 11,
                "Dec": 12,
            }
            year = getContent(publication_Date, "./Year")
            month_abbr = getContent(publication_Date, "./Month", "Jan")
            day = getContent(publication_Date, "./Day", "01")

            # Date can have alternative MedlineDate format
            if year is None:
                date = getContentUnique(publication_Date, "./MedlineDate")
                if date is not None:
                    date = date.split(" ")
                    year = date[0]
                    month_abbr = date[1] if len(date) > 1 else "Jan"
                    if len(date) > 2:
                        day = date[2].split("-")[0]
                    else:
                        day = "01"
                else:
                    return None
            month = month_abbr_to_num[month_abbr]
            return datetime.date(int(year), month, int(day))

    def _extractReferences(self, element: _Element) -> List[str] | None:

        references = []
        for reference in element.findall("PubmedData/ReferenceList/Reference"):
            references.append(getContent(reference, './/ArticleId[@IdType="doi"]', ""))
        return references if references else None

    def toJSON(self) -> str:
        """Dump the object as JSON string."""
        return json.dumps(
            {
                key: (
                    value
                    if not isinstance(value, (datetime.date, _Element))
                    else str(value)
                )
                for key, value in self.to_dict().items()
            },
            sort_keys=True,
            indent=4,
        )


@typechecked
class CrossrefArticle(Article):
    """Data class that contains a Crossref article."""

    def __init__(self, data: dict[str, Any]) -> None:
        """
        Initialize the object from a dictionary, returned by the Crossref API query.
        """

        super().__init__()
        self._extract_data(data)

    def _extract_data(self, data: dict[str, Any]) -> None:
        """Extract the data from the dictionary."""

        # some values can be directly assigned
        accept_fields = [
            "publisher",
            "issue",
            "license",
            "type",
            "volume",
            "link",
        ]
        for field in accept_fields:
            setattr(self, field, data.get(field, None))

        # others require processing
        self.title = self._extract_title(data)
        self.authors = self._extract_authors(data)
        self.is_referenced_by_count = data.get("is-referenced-by-count", None)
        self.journal = self._extract_journal(data)
        self.issn = self._extract_issn(data)
        self.start_page, self.end_page = self._extract_pages(data)
        self.references = self._extract_references(data)
        self.publication_date = self._extrac_date(data)
        self.abstract = self._extract_abstract(data)
        self.doi = data.get("DOI", None)

    def _extract_journal(self, data: dict[str, Any]) -> str | None:
        """Extract the journal from the data."""
        journal = data.get("container-title", [""])
        if len(journal) == 0 or journal[0] == "":
            return None
        return journal[0].strip()

    def _extract_title(self, data: dict[str, Any]) -> str | None:
        """Extract the title from the data."""
        title = data.get("title", [""])
        if len(title) == 0 or title[0] == "":
            return None
        # some titles contain garbage like '&lt;title&gt;' and '&lt;/title&gt;'
        # remove it
        title = re.sub(r"&lt;/?title&gt;", "", title[0])
        return title.strip()

    def _extract_authors(self, data: dict[str, Any]) -> List[dict[str, str | None]]:
        """Extract the authors from the data."""

        authors_list = data.get("author", [])
        for i in range(len(authors_list)):
            author = authors_list[i]
            author_new = {}
            if author.get("family"):
                author_new["lastname"] = author.get("family")
            else:
                author_new["lastname"] = author.get("lastname")
            if author.get("given"):
                author_new["firstname"] = author.get("given")
            else:
                author_new["firstname"] = author.get("firstname")
            affiliation = author.get("affiliation")
            if isinstance(affiliation, dict):
                author_new["affiliation"] = affiliation.get("name")
            else:
                author_new["affiliation"] = None
            authors_list[i] = author_new
        return authors_list

    def _extract_issn(self, data: dict[str, Any]) -> str | None:
        """Extract the ISSN from the data."""

        issn_list = data.get("issn-type", [])
        # get issn value in the following order: electronic, print
        for issn in issn_list:
            if issn.get("type") == "electronic":
                return issn.get("value")
        for issn in issn_list:
            if issn.get("type") == "print":
                return issn.get("value")
        return None

    def _extract_pages(self, data: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract the start and end pages from the data."""
        page = data.get("page", None)
        if page:
            pages = tuple(page.split("-"))
            if len(pages) == 2:
                return pages
            return pages[0], None
        return None, None

    def _extract_references(self, data: dict[str, Any]) -> List[str] | None:
        """Extract the references from the data."""
        references = data.get("reference", None)
        ref_list = (
            [reference.get("DOI") for reference in references if reference.get("DOI")]
            if references
            else []
        )
        return ref_list if len(ref_list) else None

    def _extrac_date(self, data: dict[str, Any]) -> datetime.date | None:
        """Extract the publication date from the data."""
        date = data.get("published", {}).get("date-parts", [[]])[0]
        if date:
            year = date[0]
            if len(date) > 1:
                month = date[1]
            else:
                month = 1
            if len(date) > 2:
                day = date[2]
            else:
                day = 1
            return datetime.date(year, month, day)

    def _extract_abstract(self, data: dict[str, Any]) -> str | None:
        """Extract the abstract from the data."""

        raw_abstract = data.get("abstract")
        if raw_abstract is not None:
            # Remove <jats:title> tags and other XML tags
            raw_abstract = re.sub(r"<jats:title>.*</jats:title>", "", raw_abstract)
            raw_abstract = re.sub(r"<[^>]+>", "", raw_abstract).strip()
            # Remove tabs and new lines
            raw_abstract = raw_abstract.replace("\t", "").replace("\n", "")
            if len(raw_abstract) > 1:
                return raw_abstract

    @classmethod
    def col_types(cls) -> Dict[str, str]:
        col_types = super().col_types()
        col_types.update({"publisher": "string"})
        return col_types

    def to_df(self) -> pd.DataFrame:
        """Convert the parsed information to a pandas DataFrame."""

        df = pd.DataFrame([self.to_dict()])
        return _format_df(df)


class ArticleCollection:
    """Class for handling a collection of articles."""

    def __init__(self, articles: Iterable[CrossrefArticle|dict]) -> None:
        """Initialize the collection with a list of articles."""
        
        self.articles = (article if isinstance(article, CrossrefArticle) else CrossrefArticle(article) for article in articles)

    def to_df(self) -> DataFrame:
        """Convert the collection to a pandas DataFrame."""
        df = pd.DataFrame([article.to_dict() for article in self.articles]) # type: ignore[assignment]
        df = _format_df(df)
        if df.size == 0:
            df = DataFrame(columns=CrossrefArticle.get_all_slots())
        return df


def load_csv(path: str) -> DataFrame:
    """
    Load a CSV file into a DataFrame.
    """

    df = pd.read_csv(path)
    df = _format_df(df)
    return df


def _format_df(df: DataFrame) -> DataFrame:
    """
    Format the DataFrame to have the correct columns and types."
    """

    cols = list(set(CrossrefArticle.get_all_slots() + PubMedArticle.get_all_slots()))
    # Ensure all columns from cols are present in the DataFrame
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    # Convert to lower case
    for col in ["title", "abstract", "authors", "journal", "publisher"]:
        df[col] = df[col].str.lower()
    # convert to python objects to python types
    for col in ["license", "link", "authors", "references"]:
        df[col] = (
            df[col].fillna("None").str.replace("none", "None").transform(literal_eval)
        )
    # apply column types
    df = df.astype(CrossrefArticle.col_types())
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    return df
