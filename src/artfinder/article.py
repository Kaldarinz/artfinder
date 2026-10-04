# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Module for handling articles."""

from __future__ import annotations

import datetime
import html
import logging
import re
from ast import literal_eval

from typing import Any, Dict, List, Iterable, Mapping

import pandas as pd
from pandas import DataFrame

from artfinder.crossref_helpers import funder_registry_id
from artfinder.dataclasses import FunderRegistryEntry

logger = logging.getLogger(__name__)

#: Columns holding lists of values or records rather than one scalar.
LIST_COLUMNS = (
    "license",
    "links",
    "authors",
    "references",
    "funders",
    "keywords",
    "issn",
)

#: Scalar columns folded to lowercase: identifiers and Crossref's vocabulary,
#: not text. Everything else keeps the case Crossref gives it.
LOWERCASE_COLUMNS = ("doi", "type")

_MARKUP_TAG = re.compile(r"</?[A-Za-z][\w.-]*(?::[\w.-]+)?\b[^<>]*>")
"""An HTML, MathML or JATS tag, as opposed to a bare `<`.

A letter-led name is required, so a size threshold such as `(<5 nm) ... (>20 nm)`
is not mistaken for one tag spanning both.
"""

_BLOCK_TAG = re.compile(r"</?(?:jats:)?(?:p|sec)\b[^<>]*>")
"""A JATS paragraph or section tag, which separates the text on either side."""


def _strip_markup(text: str) -> str:
    """
    Reduce Crossref text to plain text.

    Tags are stripped on both sides of unescaping, because some deposits escape
    their markup twice (`&lt;sub&gt;`). Whitespace, including the line breaks and
    indentation of pretty-printed XML, is collapsed to single spaces.

    Parameters
    ----------
    text : str
        Text as Crossref returns it.

    Returns
    -------
    str
        Plain text.
    """
    text = html.unescape(_MARKUP_TAG.sub("", text))
    return " ".join(_MARKUP_TAG.sub("", text).split())


# TODO: There should probably be only one Article class
class Article:
    """Base class for all articles."""

    __slots__ = (
        "title",
        "authors",
        "journal",
        "publication_date",
        "links",
        "doi",
        "type",
        "keywords",
        "citation_count",
        "abstract",
        "publisher",
        "issn",
        "volume",
        "issue",
        "start_page",
        "end_page",
        "references",
        "funders",
        "license",
    )

    def __init__(self) -> None:
        """Initialize all attributes in __slots__ to None."""
        for slot in self.__slots__:
            setattr(self, slot, None)

    def to_dict(self) -> Dict[Any, Any]:
        """
        Convert the parsed information to a Python dict.

        Scalar fields are stringified, keeping their case, except the DOI and
        type (`LOWERCASE_COLUMNS`), which are lowercased. List-valued fields
        (`LIST_COLUMNS`) are left as Python objects: their text is
        case-sensitive (URLs, ISSN check digits, names), and their repr would
        carry any None inside them as text.

        Returns
        -------
        Dict[Any, Any]
            Field names mapped to their values.
        """
        dct = {key: self.__getattribute__(key) for key in self.get_all_slots()}
        for key, val in dct.items():
            if val is None or key in LIST_COLUMNS:
                continue
            dct[key] = str(val).lower() if key in LOWERCASE_COLUMNS else str(val)
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
            A list of all __slots__ defined in the class and its superclasses,
            each once.
        """
        slots = []
        for base in cls.__mro__:  # Traverse the Method Resolution Order (MRO)
            # A subclass declaring no slots of its own inherits the attribute, so
            # read each class's own declaration only.
            for slot in base.__dict__.get("__slots__", ()):
                if slot not in slots:
                    slots.append(slot)
        return slots

    @classmethod
    def col_types(cls) -> dict[str, str]:
        """Return dictionary with column names and their types."""
        return {
            "abstract": "string",
            "title": "string",
            "doi": "string",
            "type": "string",
            "journal": "string",
            "volume": "string",
            "issue": "string",
            "start_page": "string",
            "end_page": "string",
            "citation_count": "int",
        }


class CrossrefArticle(Article):
    """Data class that contains a Crossref article."""

    def __init__(
        self,
        data: dict[str, Any],
        funder_registry: Mapping[str, FunderRegistryEntry] | None = None,
    ) -> None:
        """
        Initialize the object from a dictionary, returned by the Crossref API query.

        Parameters
        ----------
        data : dict[str, Any]
            Crossref work record.
        funder_registry : Mapping[str, FunderRegistryEntry] | None, optional
            Funder Registry entries by funder id. A funder found here is named
            as the registry names it rather than as the publisher deposited it.
        """

        super().__init__()
        self._extract_data(data, funder_registry or {})

    def _extract_data(
        self,
        data: dict[str, Any],
        funder_registry: Mapping[str, FunderRegistryEntry],
    ) -> None:
        """Extract the data from the dictionary."""

        # some values can be directly assigned
        self.publisher = data.get("publisher", None)
        self.issue = data.get("issue", None)
        self.license = data.get("license", None)
        self.type = data.get("type", None)
        self.volume = data.get("volume", None)

        # others require processing
        self.title = self._extract_title(data)
        self.authors = self._extract_authors(data)
        self.citation_count = data.get("is-referenced-by-count", None)
        self.journal = self._extract_journal(data)
        self.issn = self._extract_issn(data)
        self.start_page, self.end_page = self._extract_pages(data)
        self.references = self._extract_references(data)
        self.publication_date = self._extrac_date(data)
        self.abstract = self._extract_abstract(data)
        self.doi = data.get("DOI", None)
        self.funders = self._extract_funder(data, funder_registry)
        self.links = self._extract_link(data)
        self.keywords = self._extract_keywords(data)

    def _extract_keywords(self, data: dict[str, Any]) -> list[str]:
        """Extract the keywords from the data."""
        keywords = data.get("subject", [])
        return [keyword.strip().lower() for keyword in keywords if keyword.strip()]

    def _extract_link(self, data: dict[str, Any]) -> List[dict[str, str | None]]:
        """Extract the link from the data."""

        link_list_raw = data.get("link", [])
        link_list = []
        for link in link_list_raw:
            link_new = {}
            if link.get("URL"):
                link_new["url"] = link.get("URL")
            if link.get("content-type"):
                link_new["content_type"] = link.get("content-type")
            if len(link_new):
                link_list.append(link_new)
        return link_list

    def _extract_funder(
        self,
        data: dict[str, Any],
        funder_registry: Mapping[str, FunderRegistryEntry],
    ) -> list[dict[str, str | list[str]]]:
        """
        Extract the funders from the data, one record per award.

        A publisher can deposit a funder's name with every non-ASCII character
        replaced by `?`. A funder whose DOI has an entry in `funder_registry`
        therefore takes its name, and its alternative names, from that entry;
        any other keeps the name as deposited.

        A Crossref funder record may list several awards; each becomes a record
        of its own carrying the funder's names and `doi` and one `number`, in
        the order Crossref lists them. A funder with no awards gives a single
        record without `number`. Award strings are kept exactly as deposited.

        Parameters
        ----------
        data : dict[str, Any]
            Crossref work record.
        funder_registry : Mapping[str, FunderRegistryEntry]
            Funder Registry entries by funder id.

        Returns
        -------
        list[dict[str, str | list[str]]]
            One record per award, with whichever of `name`, `alt_names`, `doi`
            and `number` it has.
        """

        funder_list: list[dict[str, str | list[str]]] = []
        for funder in data.get("funder", []):
            entry = funder_registry.get(funder_registry_id(funder.get("DOI")) or "")
            awards: list[str | None] = list(funder.get("award") or []) or [None]
            for award in awards:
                record: dict[str, str | list[str]] = {}
                if entry is not None:
                    record["name"] = entry.name
                    if entry.alt_names:
                        # A list of its own per record, not one shared by the awards.
                        record["alt_names"] = list(entry.alt_names)
                elif funder.get("name"):
                    record["name"] = funder["name"]
                if funder.get("DOI"):
                    record["doi"] = funder["DOI"]
                if award is not None:
                    record["number"] = award
                funder_list.append(record)
        return funder_list

    def _extract_journal(self, data: dict[str, Any]) -> str | None:
        """Extract the journal from the data."""
        journal = data.get("container-title", [""])
        if len(journal) == 0 or journal[0] == "":
            return None
        return journal[0].strip().replace("&amp;", "and")

    def _extract_title(self, data: dict[str, Any]) -> str | None:
        """
        Extract the title from the data, as plain text.

        Crossref titles can carry markup (`<sub>`, MathML, a stray escaped
        `&lt;title&gt;`) and the line breaks of pretty-printed XML.

        Parameters
        ----------
        data : dict[str, Any]
            Crossref work record.

        Returns
        -------
        str | None
            The title, or None when the record has none.
        """
        titles = data.get("title") or [""]
        return _strip_markup(titles[0]) or None

    def _extract_authors(
        self, data: dict[str, Any]
    ) -> List[dict[str, str | list[str] | None]]:
        """
        Extract the authors from the data, in the order Crossref lists them.

        Crossref also lists organisations as authors, each a bare `name` with no
        `family` or `given`. Only people listed outside an organisation's section
        are authors here (`_drop_organisation_sections`). Every one of them is
        kept, even one listed twice: two entries with the same name may be two
        people.

        Parameters
        ----------
        data : dict[str, Any]
            Crossref work record.

        Returns
        -------
        List[dict[str, str | list[str] | None]]
            One record per author.
        """

        authors_list = self._drop_organisation_sections(
            data.get("author", []), data.get("DOI")
        )
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
            if isinstance(affiliation, list):
                author_new["affiliation"] = [
                    aff.get("name") for aff in affiliation if aff.get("name")
                ]
            if orcid := author.get("ORCID"):
                author_new["orcid"] = orcid.split("/")[-1]
            if author.get("sequence"):
                if i == len(authors_list) - 1:
                    author_new["position"] = "last"
                else:
                    author_new["position"] = author.get("sequence")
            elif i == 0:
                author_new["position"] = "first"
            elif i == len(authors_list) - 1:
                author_new["position"] = "last"
            else:
                author_new["position"] = "additional"
            authors_list[i] = author_new
        return authors_list

    @staticmethod
    def _drop_organisation_sections(
        authors: list[dict[str, Any]], doi: str | None
    ) -> list[dict[str, Any]]:
        """
        Keep the people listed before the first organisation in an author array.

        An organisation is an entry with neither a family nor a given name. It
        opens a section of the byline, and the people listed after it are its
        members, not authors in their own right. Crossref's array is flat, with
        nothing marking where a section ends, so the organisation and everything
        after it is dropped.

        Parameters
        ----------
        authors : list[dict[str, Any]]
            Crossref `author` array.
        doi : str | None
            DOI of the work, for the log.

        Returns
        -------
        list[dict[str, Any]]
            The people before the first organisation, in Crossref's order.
        """

        for index, author in enumerate(authors):
            if not (
                author.get("family")
                or author.get("lastname")
                or author.get("given")
                or author.get("firstname")
            ):
                logger.info(
                    f"{doi}: dropped organisation {author.get('name')!r} and the "
                    f"{len(authors) - index - 1} entries after it, keeping {index} "
                    "authors."
                )
                return authors[:index]
        return list(authors)

    def _extract_issn(self, data: dict[str, Any]) -> list[str]:
        """Extract the ISSN from the data."""

        issn_list_raw = data.get("issn-type", [])
        issn_list = []
        # get issn value in the following order: electronic, print
        for issn in issn_list_raw:
            issn_list.append(re.sub(r"\W+", "", issn.get("value", "")))
        return issn_list

    def _extract_pages(self, data: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract the start and end pages from the data."""
        page = data.get("page", None)
        if page:
            pages = tuple(page.split("-"))
            if len(pages) == 2:
                return pages
            return pages[0], None
        return None, None

    def _extract_references(self, data: dict[str, Any]) -> List[str]:
        """Extract the lowercased DOIs of the references from the data."""
        references = data.get("reference", None)
        ref_list = (
            [
                reference.get("DOI").lower()
                for reference in references
                if reference.get("DOI")
            ]
            if references
            else []
        )
        return ref_list

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
        return None

    def _extract_abstract(self, data: dict[str, Any]) -> str | None:
        """
        Extract the abstract from the data, as plain text.

        Section titles (`<jats:title>`) are dropped with their text, paragraphs
        and sections are separated by a space, and the rest is cleaned as a
        title is (`_strip_markup`).

        Parameters
        ----------
        data : dict[str, Any]
            Crossref work record.

        Returns
        -------
        str | None
            The abstract, or None when the record has none.
        """

        raw_abstract = data.get("abstract")
        if raw_abstract is None:
            return None
        raw_abstract = re.sub(
            r"<jats:title>.*?</jats:title>", "", raw_abstract, flags=re.DOTALL
        )
        abstract = _strip_markup(_BLOCK_TAG.sub(" ", raw_abstract))
        return abstract if len(abstract) > 1 else None

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

    def __init__(self, articles: Iterable[CrossrefArticle | dict]) -> None:
        """Initialize the collection with a list of articles."""

        self.articles = (
            (
                article
                if isinstance(article, CrossrefArticle)
                else CrossrefArticle(article)
            )
            for article in articles
        )

    def to_df(self) -> DataFrame:
        """Convert the collection to a pandas DataFrame."""
        df = pd.DataFrame([article.to_dict() for article in self.articles])
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


def _parse_list_value(value: Any) -> object:
    """
    Turn one cell of a list-valued column into a Python object.

    An article converted in memory already holds the list; a CSV holds its repr,
    which is parsed back.

    Parameters
    ----------
    value : Any
        Cell value: a list or dict, its repr, or a missing value.

    Returns
    -------
    object
        The list or dict, or None for a missing value.
    """
    if isinstance(value, str):
        return literal_eval(value)
    if isinstance(value, (list, dict)) or not pd.isna(value):
        return value
    return None


def _format_df(df: DataFrame) -> DataFrame:
    """
    Format the DataFrame to have the correct columns and types."
    """

    cols = list(set(CrossrefArticle.get_all_slots()))
    # Ensure all columns from cols are present in the DataFrame
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    for col in LIST_COLUMNS:
        df[col] = df[col].map(_parse_list_value)
    # apply column types
    df = df.astype(CrossrefArticle.col_types())
    df["publication_date"] = pd.to_datetime(
        df["publication_date"], errors="coerce"
    ).dt.date
    return df
