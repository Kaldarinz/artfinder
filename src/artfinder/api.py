# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""API module."""

from __future__ import annotations

import warnings
import logging
from typing import (
    Any,
    Dict,
    Generator,
    cast,
    Callable,
    ParamSpec,
    TypeVar,
    Coroutine,
    Literal,
    Self,
)
import re
from datetime import datetime
from pathlib import Path
import os

import pandas as pd
from pandas import DataFrame, Series

from artfinder.article import CrossrefArticle
from artfinder.crossref import Crossref
from artfinder.dataclasses import DocumentType
from artfinder.http_requests import FileDownloader

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class ArtFinder:
    """
    Base class for ArtFinder API.
    """

    def __init__(self, email: str | None = None) -> None:
        """
        Initialize the ArtFinder object.

        Returns
        -------
        None
        """
        self.email = email

    def find_article(
        self,
        *,
        doi: str | None = None,
        title: str | None = None,
        database: Literal["pubmed", "crossref", "all"] = "crossref",
    ) -> Series:
        """
        Use this to get single article by title or doi.

        Doi will return exactly requested article, while title will return the first
        mathing article (title will *always* return some result).

        Parameters
        ----------
        doi : str | None
            DOI of the article to find.
        title : str | None
            Title of the article to find.
        database : Literal["pubmed", "crossref", "all"]
            Database to search in.

        Returns
        -------
        Series containing the article information.
        """

        if doi is None and title is None:
            raise ValueError("Either DOI or title must be provided.")
        if database in ["pubmed", "all"]:
            raise NotImplementedError("Only crossref support is implemented.")

        if title is not None:
            df = (
                Crossref(email=self.email).search(title).article().get_df(max_results=1)
            )
        else:
            df = Crossref(email=self.email).doi(doi)  # type: ignore
        return pd.Series(df.iloc[0]) if not df.empty else pd.Series(index=df.columns)

    def search(
        self,
        query: str | None = None,
        author: str | None = None,
        pub_since: str | datetime | None = None,
        pub_until: str | datetime | None = None,
        database: Literal["pubmed", "crossref", "all"] = "crossref",
        max_results: int | None = 100,
    ) -> DataFrame:
        """
        Search for articles.

        Parameters
        ----------
        query : str | None
            Search query. Can be a title or other search term.
            Can also be None if some other field is provided.
        author : str | None
            Author name to search for. Query field can also contain author name,
            but this is a strict filter for author name.
        pub_since : str | datetime | None
            Publication date since which to search for articles.
            Can be a string in YYYY, YYYY-MM, YYYY-MM-DD format or a datetime object.
        pub_until : str | datetime | None
            Publication date until which to search for articles.
            Can be a string in YYYY, YYYY-MM, YYYY-MM-DD format or a datetime object.
        database : Literal["pubmed", "crossref", "all"]
            Database to search in. Can be "pubmed", "crossref" or "all".
        max_results : int | None
            Maximum number of results to return. If None, all results are returned.
            It is better to check the number of results first using isearch() method

        Returns
        -------
        DataFrame containing the search results.
        """

        if database != "crossref":
            raise NotImplementedError("Only crossref support is implemented.")
        if not any([query, author, pub_since, pub_until]):
            raise ValueError(
                "At least one of query, author, pub_since or pub_until must be provided."
            )
        return (
            Crossref(email=self.email)
            .search(query)
            .author(author)
            .filter(from_pub_date=pub_since)
            .filter(until_pub_date=pub_until)
            .article()
            .get_df(max_results=max_results)
        )

    def isearch(
        self,
        query: str | None = None,
        author: str | None = None,
        pub_since: str | datetime | None = None,
        pub_until: str | datetime | None = None,
        database: Literal["pubmed", "crossref", "all"] = "crossref",
    ) -> int:
        """
        Get number of articles, which comply search.

        Parameters
        ----------
        query : str | None
            Search query. Can be a title or other search term.
            Can also be None if some other field is provided.
        author : str | None
            Author name to search for. Query field can also contain author name,
            but this is a strict filter for author name.
        pub_since : str | datetime | None
            Publication date since which to search for articles.
            Can be a string in YYYY, YYYY-MM, YYYY-MM-DD format or a datetime object.
        pub_until : str | datetime | None
            Publication date until which to search for articles.
            Can be a string in YYYY, YYYY-MM, YYYY-MM-DD format or a datetime object.
        database : Literal["pubmed", "crossref", "all"]
            Database to search in. Can be "pubmed", "crossref" or "all".
        max_results : int | None
            Maximum number of results to return. If None, all results are returned.
            It is better to check the number of results first using isearch() method

        Returns
        -------
        Number of articles, which comply search term.
        """

        if database != "crossref":
            raise NotImplementedError("Only crossref support is implemented.")
        if not any([query, author, pub_since, pub_until]):
            raise ValueError(
                "At least one of query, author, pub_since or pub_until must be provided."
            )
        return (
            Crossref(email=self.email)
            .search(query)
            .author(author)
            .filter(from_pub_date=pub_since)
            .filter(until_pub_date=pub_until)
            .article()
            .count()
        )

    def get_refs(self, articles: CrossrefArticle | Series | DataFrame) -> DataFrame:
        """
        Get articles, cited in the given articles.

        Parameters
        ----------
        article : CrossrefArticle|Series|DataFrame
            Article(s) to get citing articles for.
            Series and DataFrame must contain a column with named "references".

        Returns
        -------
        DataFrame containing the cited articles.
        """

        dois = []
        # Create list of dois for referenced articles
        if isinstance(articles, CrossrefArticle):
            if (more := articles.references) is not None:
                dois.extend(more)
        elif isinstance(articles, Series):
            if (more := articles["references"]) is not None:
                dois.extend(more)
        elif isinstance(articles, DataFrame):
            for _, article in articles.iterrows():
                if (more := article["references"]) is not None:  # type: ignore
                    dois.extend(more)
        else:
            raise TypeError("article must be CrossrefArticle, Series or DataFrame")

        dois = list(set(dois))
        return Crossref(email=self.email).get_dois(dois)

    def download_pdf(
        self,
        articles: CrossrefArticle | Series | DataFrame,
        path: str | None = None,
        name: Literal["doi", "title"] = "title",
        max_connections: int = 5,
    ) -> FileDownloader:
        """
        Download pdf files for the given articles.

        Parameters
        ----------
        articles : CrossrefArticle|Series|DataFrame
            Article(s) to download pdf files for.
            Series and DataFrame must contain a column with named "doi".
        path : str | None
            Path to save the pdf files. If None, pdfs folder will be
            created in the current folder.
        name : Literal["doi", "title"]
            Name of the file. Can be either "doi" or "title".
        max_connections : int
            Maximum number of connections to use for downloading.

        Returns
        -------
        None
        """

        if isinstance(articles, CrossrefArticle):
            articles = articles.to_df()
        elif isinstance(articles, Series):
            articles = pd.DataFrame([articles])
        elif not isinstance(articles, DataFrame):
            raise TypeError("article must be CrossrefArticle, Series or DataFrame")

        dois = articles["doi"].tolist()
        titles = articles["title"].tolist()
        if path is None:
            path = os.path.join(os.getcwd(), "pdfs")
        # create directory if it does not exist
        Path(path).mkdir(parents=True, exist_ok=True)

        # Regular expression to replace invalid characters
        invalid_chars = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
        paths = [
            os.path.join(
                path,
                invalid_chars.sub(
                    "_", f"{doi}.pdf" if name == "doi" else f"{title}.pdf"
                ),
            )
            for doi, title in zip(dois, titles)
        ]
        return FileDownloader(
            links=articles["link"].to_list(),
            save_paths=paths,
            concurency_limit=max_connections,
        ).download_files()
