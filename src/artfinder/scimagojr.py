# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
Classes for handling SciMagoJR data.
"""

import logging
import re
from pathlib import Path
from typing import cast

from numpy import true_divide
import pandas as pd
from pyiso4.ltwa import Abbreviate

logger = logging.getLogger(__name__)


class SciMagoJR:
    """Journals data from SciMagoJR."""

    BASE_PATH = Path(__file__).parent.parent.parent / "data"

    def __init__(self, ranking_year: str = "latest"):

        self.all_data: pd.DataFrame
        if ranking_year == "latest":
            # Find the latest year available
            years = [d for d in self.BASE_PATH.glob("scimagojr_*")]
            years.sort(key=lambda x: x.name, reverse=True)
            self.all_data = pd.read_csv(years[0], header=0, sep=";", decimal=",")
        else:
            raise NotImplementedError(
                "Only 'latest' ranking year is supported currently."
            )

    def get_journal(
        self, title: str | None = None, issn: str | None = None
    ) -> pd.Series | None:
        """
        Get journal data for a given title or ISSN.

        Parameters
        -----------
            title: The title of the journal.
            issn: The ISSN of the journal. Should be in format '12345678' or '12345678X'

        Returns
        -------
            A pandas Series containing the journal data, or None if not found.
        """

        if title is None and issn is None:
            logger.error("Either title or issn must be provided.")
            return None

        journal_data = pd.DataFrame()
        if title is not None:
            title = title.replace("&amp;", "and")
            journal_data = self.all_data[
                self.all_data["Title"].str.lower() == title.lower()
            ].copy()
            if not journal_data.empty:
                journal_data.loc[:, "title"] = title
        if journal_data.empty and not pd.isna(issn):  # type: ignore
            journal_data = self.all_data[self.all_data["Issn"] == issn]
        if journal_data.empty:
            logger.error("Journal not found in SciMagoJR data.")
            return None

        journal_data = journal_data[
            [
                "Title",
                "Type",
                "Issn",
                "Publisher",
                "Open Access",
                "SJR Best Quartile",
                "Citations / Doc. (2years)",
                "Rank",
                "Country",
                "Categories",
                "Areas",
            ]
        ]
        journal_data = journal_data.reset_index(drop=True).iloc[0]  # type: ignore
        journal_data.index = [
            "title",
            "type",
            "issns",
            "publisher",
            "open_access",
            "quartile",
            "impact_factor",
            "rank",
            "country",
            "categories",
            "areas",
        ]
        # Add abbreviation
        journal_data["abbreviation"] = Abbreviate.create()(
            title=journal_data.title, remove_part=True
        )

        # Foramt open_access
        journal_data["open_access"] = {"Yes": True, "No": False}.get(
            journal_data["open_access"], False
        )

        # Format Categories
        cats = journal_data["categories"].replace(";", ",")
        journal_data["categories"] = re.sub(r"\s*\(Q\d\)", "", cats)
        return journal_data
