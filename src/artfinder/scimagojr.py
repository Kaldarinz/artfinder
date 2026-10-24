# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
Classes for handling SciMagoJR data.
"""

import logging
import re
from pathlib import Path

import pandas as pd
from pyiso4.ltwa import Abbreviate  # type: ignore[import-untyped]

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
        self.publisher_countries = self.publisher_country_map(self.all_data)

    @staticmethod
    def publisher_country_map(data: pd.DataFrame) -> dict[str, str]:
        """
        Map each publisher to the country most of its journals give.

        SciMagoJR's "Country" is the journal's country, which varies across one
        publisher's journals: most Elsevier B.V. journals give the Netherlands, a
        few give China. The publisher's country is taken as the most common
        "Country" over the rows with exactly that "Publisher". A tie goes to the
        country that sorts first alphabetically. Rows without a publisher or a
        country are ignored.

        Parameters
        ----------
        data
            SciMagoJR data with "Publisher" and "Country" columns.

        Returns
        -------
        dict[str, str]
            Country of each publisher, keyed by the "Publisher" value.
        """
        counts = (
            data.dropna(subset=["Publisher", "Country"])
            .groupby(["Publisher", "Country"])
            .size()
            .reset_index(name="journals")
        )
        modes = counts.sort_values(
            ["Publisher", "journals", "Country"], ascending=[True, False, True]
        ).drop_duplicates("Publisher")
        return dict(zip(modes["Publisher"], modes["Country"]))

    def get_journal(
        self, title: str | None = None, issn: list[str] | None = None
    ) -> pd.Series | None:
        """
        Get journal data for a given title or ISSN.

        Parameters
        ----------
        title
            The title of the journal.
        issn
            ISSNs of the journal, each in the format '12345678' or '1234567X'.

        Returns
        -------
        pd.Series | None
            The journal data, or None if not found. "country" is the publisher's
            country (see `publisher_country_map`), not SciMagoJR's per-journal
            "Country"; a journal without a publisher keeps its own.
        """

        # Callers pass values straight out of a DataFrame, where an absent title is
        # pd.NA or NaN and an absent ISSN list is empty. Treat both as "not provided"
        # rather than calling string methods on them.
        if not isinstance(title, str) or not title.strip():
            title = None
        issn = [i.strip() for i in issn or [] if isinstance(i, str) and i.strip()] or None

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
        if journal_data.empty and issn is not None:
            for issn_ in issn:
                # The ISSN is data, not a pattern -- a stray "(" would raise.
                journal_data = self.all_data[
                    self.all_data["Issn"].str.contains(re.escape(issn_))
                ]
                if not journal_data.empty:
                    break
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
        journal = journal_data.reset_index(drop=True).iloc[0]
        journal.index = [
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
        journal["country"] = self.publisher_countries.get(
            journal["publisher"], journal["country"]
        )
        # Add abbreviation
        journal["abbreviation"] = Abbreviate.create()(
            title=journal.title, remove_part=True
        )

        # Foramt open_access
        journal["open_access"] = {"Yes": True, "No": False}.get(
            journal["open_access"], False
        )

        # Format Categories
        cats = journal["categories"].replace(";", ",")
        journal["categories"] = re.sub(r"\s*\(Q\d\)", "", cats)

        # Format areas
        journal["areas"] = journal["areas"].replace(";", ",")
        return journal
