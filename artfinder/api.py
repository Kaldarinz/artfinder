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
from pandas import DataFrame

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
        doi: str | None = None,
        title: str | None = None,
        database: Literal["pubmed", "crossref", "all"] = "crossref",
    ) -> DataFrame:
        """
        Find an article by DOI or title.
        """

        if doi is None and title is None:
            raise ValueError("Either DOI or title must be provided.")
        if database in ["pubmed", "all"]:
            raise NotImplementedError("only crossref is supported for now")

        if title is not None:
            return self.cr.query(
                bibliographic=re.sub(r"\W+", "+", title.strip())
            ).get_df()
        raise NotImplementedError("function not implemented yet")





