# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
Classes for handling SciMagoJR data.
"""

from pathlib import Path
import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

class SciMagoJR:
    """Class for handling SciMagoJR data."""

    BASE_PATH = Path(__file__).parent.parent.parent / 'data'

    def __init__(self, ranking_year: str = 'latest'):

        self.all_data: pd.DataFrame
        if ranking_year == 'latest':
            # Find the latest year available
            years = [d for d in self.BASE_PATH.glob('scimagojr_*')]
            years.sort(key=lambda x: x.name, reverse=True)
            self.all_data = pd.read_csv(years[0], header=0, sep=';', decimal=',')
        else:
            raise NotImplementedError("Only 'latest' ranking year is supported currently.")
        
    def get_ranking(self, title: str | None = None, issn: str | None = None) -> pd.Series | None:
        """Get the ranking for a given journal title or ISSN."""
        journal_data = pd.DataFrame()
        if title is None and issn is None:
            logger.error("Either title or issn must be provided.")
            return None
        if title is not None:
            title = title.replace("&amp;", "and")
            journal_data = self.all_data[self.all_data['Title'].str.lower() == title.lower()].copy()
            if not journal_data.empty:
                journal_data.loc[:, 'title'] = title
        if journal_data.empty and not pd.isna(issn):
            issn = re.sub(r'\D', '', issn)
            journal_data = self.all_data[self.all_data['Issn'] == issn]
        if journal_data.empty:
            logger.error("Journal not found in SciMagoJR data.")
            return None
    
        journal_data = journal_data[['Title', 'Type', 'Issn', 'Publisher', 'Open Access', 'SJR Best Quartile', 'Citations / Doc. (2years)', 'Country']]
        journal_data.columns = ['title', 'type', 'issn', 'publisher', 'open_access', 'quartile', 'if_2_years', 'country']
        journal_data['open_access'] = journal_data['open_access'].map({'Yes': '1', 'No': '0'})
        return journal_data.reset_index(drop=True).iloc[0]
