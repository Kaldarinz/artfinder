"""
Tests for journal lookup when article metadata is missing or incomplete.
"""

from typing import Any

import pandas as pd
import pytest

from artfinder.api import ArtFinder, _clean_issn, _clean_str
from artfinder.scimagojr import SciMagoJR

# A journal that is present in the SciMagoJR data under both title and ISSN.
KNOWN_TITLE = "Scientific Reports"
KNOWN_ISSN = "20452322"


def _article(
    *,
    journal: Any,
    issn: Any,
    type_: str = "journal-article",
    publisher: str = "nature publishing group",
) -> pd.Series:
    """
    Build an article Series of the shape get_journal_info reads.

    Parameters
    ----------
    journal : Any
        Value for the journal field.
    issn : Any
        Value for the issn field.
    type_ : str
        Value for the type field.
    publisher : str
        Value for the publisher field.

    Returns
    -------
    pd.Series
        Article-like Series.
    """

    return pd.Series(
        {"journal": journal, "issn": issn, "type": type_, "publisher": publisher},
        dtype=object,
    )


class TestCleanHelpers:
    """Tests for the missing-value helpers at the ArtFinder boundary."""

    @pytest.mark.parametrize("missing", [None, pd.NA, float("nan"), "", "   "])
    def test_missing_titles_become_none(self, missing: Any) -> None:
        """Every flavour of absent title normalizes to None."""
        assert _clean_str(missing) is None

    def test_real_title_is_stripped(self) -> None:
        """A usable title survives, without surrounding whitespace."""
        assert _clean_str("  Scientific Reports ") == "Scientific Reports"

    @pytest.mark.parametrize("missing", [None, pd.NA, [], ["", "  "]])
    def test_missing_issns_become_none(self, missing: Any) -> None:
        """An empty or all-blank ISSN collection normalizes to None."""
        assert _clean_issn(missing) is None

    def test_issn_list_is_cleaned(self) -> None:
        """Blank entries are dropped and the rest are stripped."""
        assert _clean_issn([" 20452322 ", ""]) == ["20452322"]

    def test_bare_issn_string_is_wrapped(self) -> None:
        """A scalar ISSN is accepted as a one-element list."""
        assert _clean_issn("20452322") == ["20452322"]


class TestSciMagoMissingValues:
    """SciMagoJR.get_journal must not call string methods on missing values."""

    def test_na_title_and_empty_issn_returns_none(self) -> None:
        """The reported crash case returns None instead of raising."""
        journal = SciMagoJR("latest").get_journal(
            title=pd.NA,  # type: ignore[arg-type]
            issn=[],
        )
        assert journal is None

    def test_na_title_falls_back_to_issn(self) -> None:
        """A missing title does not prevent the ISSN lookup."""
        journal = SciMagoJR("latest").get_journal(
            title=pd.NA,  # type: ignore[arg-type]
            issn=[KNOWN_ISSN],
        )
        assert journal is not None
        assert isinstance(journal.title, str)

    def test_empty_issn_falls_back_to_title(self) -> None:
        """An empty ISSN list does not prevent the title lookup."""
        journal = SciMagoJR("latest").get_journal(title=KNOWN_TITLE, issn=[])
        assert journal is not None
        assert journal.title == KNOWN_TITLE

    def test_issn_is_not_treated_as_regex(self) -> None:
        """A malformed ISSN is matched literally rather than compiled."""
        assert SciMagoJR("latest").get_journal(title=None, issn=["2045(2322"]) is None


class TestGetJournalInfo:
    """Tests for ArtFinder.get_journal_info with incomplete articles."""

    def test_na_journal_title_uses_issn(self) -> None:
        """A pd.NA title is tolerated when the ISSN is usable."""
        finder = ArtFinder(print_status=False)
        journal = finder.get_journal_info(
            article=_article(journal=pd.NA, issn=[KNOWN_ISSN])
        )
        assert journal is not None
        assert isinstance(journal.title, str)

    def test_empty_issn_uses_title(self) -> None:
        """An empty ISSN list is tolerated when the title is usable."""
        finder = ArtFinder(print_status=False)
        journal = finder.get_journal_info(article=_article(journal=KNOWN_TITLE, issn=[]))
        assert journal is not None
        assert journal.title == KNOWN_TITLE

    def test_both_missing_returns_none(self) -> None:
        """The reported crash case returns None instead of raising."""
        finder = ArtFinder(print_status=False)
        assert finder.get_journal_info(article=_article(journal=pd.NA, issn=[])) is None

    def test_nan_journal_title_returns_none(self) -> None:
        """A NaN title from a DataFrame round-trip is handled like pd.NA."""
        finder = ArtFinder(print_status=False)
        article = _article(journal=float("nan"), issn=[])
        assert finder.get_journal_info(article=article) is None

    def test_spie_override_still_applies(self) -> None:
        """SPIE proceedings keep their hardcoded ISSNs despite a missing one."""
        finder = ArtFinder(print_status=False)
        article = _article(
            journal=pd.NA, issn=[], type_="proceedings-article", publisher="spie"
        )
        journal = finder.get_journal_info(article=article)
        assert journal is not None


class TestFindArticleNotFound:
    """find_article must not present a missing article as usable metadata."""

    def test_empty_result_is_an_empty_series(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Crossref miss yields an empty Series and a warning, not an all-NA row."""
        columns = ["doi", "journal", "issn", "type", "publisher"]

        class _StubCrossref:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def doi(self, doi: str) -> pd.DataFrame:
                return pd.DataFrame(columns=columns)

        monkeypatch.setattr("artfinder.api.Crossref", _StubCrossref)
        finder = ArtFinder(print_status=False)
        with pytest.warns(UserWarning, match="No article found"):
            result = finder.find_article(doi="10.1073/pnas")
        assert result.empty

    def test_empty_result_is_safe_for_journal_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The empty result no longer crashes the downstream journal lookup."""
        columns = ["doi", "journal", "issn", "type", "publisher"]

        class _StubCrossref:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def doi(self, doi: str) -> pd.DataFrame:
                return pd.DataFrame([dict.fromkeys(columns, pd.NA)], columns=columns)

        monkeypatch.setattr("artfinder.api.Crossref", _StubCrossref)
        finder = ArtFinder(print_status=False)
        article = finder.find_article(doi="10.1073/pnas")
        assert finder.get_journal_info(article=article) is None
