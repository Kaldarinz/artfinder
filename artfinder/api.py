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
import pandas as pd
from pandas import DataFrame, Series

from artfinder.article import PubMedArticle, CrossrefArticle
from artfinder.crossref import Crossref

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
        self.cr = Crossref(email=email)

    def find_article(
        self,
        *,
        doi: str | None = None,
        title: str | None = None,
        database: Literal["pubmed", "crossref", "all"] = "crossref",
    ) -> Series:
        """
        Use this to get single article by title or doi.
        """

        if doi is None and title is None:
            raise ValueError("Either DOI or title must be provided.")
        if database in ["pubmed", "all"]:
            raise NotImplementedError("Only crossref support is implemented.")

        if title is not None:
            # IDK, probably just self.cr.search(title) works as well
            """ return self.cr.search(
                re.sub(r"\W+", "+", title.strip()), max_results
            ).get_df() """
            df = self.cr.search(title, max_results=1).get_df()
        else:
            df = self.cr.doi(doi) # type: ignore
        return pd.Series(df.iloc[0]) if not df.empty else pd.Series(index=df.columns)
    
    def get_refd(self, article: CrossrefArticle|Series|DataFrame) -> DataFrame:
        """
        Get cited articles for given articles.
        
        Parameters
        ----------
        article : CrossrefArticle|Series|DataFrame
            Article(s) to get citing articles for.
        """

        if isinstance(article, CrossrefArticle):
            doi = article.doi
        elif isinstance(article, Series):
            doi = article["doi"]
        elif isinstance(article, DataFrame):
            doi = article.iloc[0]["doi"]
        else:
            raise TypeError("article must be CrossrefArticle, Series or DataFrame")
        
        


