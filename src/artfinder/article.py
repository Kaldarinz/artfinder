# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Module for handling articles."""

from __future__ import annotations

import datetime
import re
from ast import literal_eval

from typing import Any, Dict, List, Iterable

import pandas as pd
from pandas import DataFrame



# TODO: There should probably be only one Article class
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

    cols = list(set(CrossrefArticle.get_all_slots()))
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
