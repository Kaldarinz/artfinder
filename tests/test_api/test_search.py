"""
Tests for `ArtFinder.search` as a lookup of candidate articles by title.
"""

import copy
import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from artfinder import SearchError
from artfinder.api import ArtFinder
from artfinder.crossref import Crossref
from artfinder.http_requests import AsyncHTTPRequest

WORK_PATH = (
    Path(__file__).parents[1]
    / "test_crossref"
    / "data"
    / "work_10.1021_acsnano.5c00546.json"
)
COLUMNS = ["title", "authors", "journal", "publication_date", "doi"]
ROYAL_SOCIETY_DOI = "10.13039/501100000288"
DEADLINE = 30.0
"Seconds a call running request threads may take before it counts as hung."


def _work() -> dict[str, Any]:
    """
    Read the saved Crossref record of the ACS Nano work.

    Returns
    -------
    dict[str, Any]
        The work record.
    """

    with open(WORK_PATH, encoding="utf-8") as file:
        return json.load(file)["message"]


def _answer(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any] | None,
    funder_body: dict[str, Any] | None = None,
) -> None:
    """
    Answer every Crossref request with one response body.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest's monkeypatch fixture.
    body : dict[str, Any] | None
        Response body, or None for a failed request.
    funder_body : dict[str, Any] | None, optional
        Response body of every Funder Registry request; None, the default, for a
        failed lookup.
    """

    def async_get(urls: list[str] | str, **kwargs: Any) -> dict[str, dict | None]:
        urls = [urls] if isinstance(urls, str) else urls
        return {
            url: copy.deepcopy(funder_body if "/funders/" in url else body)
            for url in urls
        }

    monkeypatch.setattr(AsyncHTTPRequest, "async_get", staticmethod(async_get))
    monkeypatch.setattr(Crossref, "_funder_registry", {})


def _stub_get(
    monkeypatch: pytest.MonkeyPatch, error: BaseException | None = None
) -> None:
    """
    Replace the request coroutine, leaving the thread that runs it in place.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest's monkeypatch fixture.
    error : BaseException | None, optional
        Exception for the coroutine to raise; None, the default, for every request
        to fail the way `_get` reports a failure.
    """

    async def _get(
        self: AsyncHTTPRequest,
        urls: list[str],
        params: dict[str, str] | None,
        only_headers: bool,
        timeout: int,
        print_progress: bool,
    ) -> dict[str, dict | None]:
        if error is not None:
            raise error
        return {url: None for url in urls}

    monkeypatch.setattr(AsyncHTTPRequest, "_get", _get)


def _raised_within_deadline(call: Callable[[], object]) -> BaseException | None:
    """
    Run a call in a daemon thread and return what it raised.

    A call that never returns fails the test instead of stalling the suite.

    Parameters
    ----------
    call : Callable[[], object]
        Call to run.

    Returns
    -------
    BaseException | None
        What the call raised, or None if it returned.
    """

    raised: list[BaseException] = []

    def run() -> None:
        try:
            call()
        except BaseException as error:  # noqa: BLE001 - handed to the test
            raised.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=DEADLINE)
    assert not thread.is_alive(), f"The call did not return within {DEADLINE} s."
    return raised[0] if raised else None


def test_hits_have_candidate_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hit carries what a list of candidates shows."""
    _answer(monkeypatch, {"message": {"items": [_work()]}})

    df = ArtFinder(print_status=False).search(query="van der waals", max_results=5)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["doi"] == "10.1021/acsnano.5c00546"
    assert row["title"] == "Tunable Nanostructuring for van der Waals Materials"
    assert row["journal"] == "ACS Nano"
    assert row["authors"][0]["lastname"] == "Tselikov"
    assert str(row["publication_date"]) == "2025-06-16"


def test_no_hits_give_empty_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """No hits give an empty frame with each column once."""
    _answer(monkeypatch, {"message": {"items": []}})

    df = ArtFinder(print_status=False).search(query="nothing", max_results=5)

    assert df.empty
    assert df.columns.is_unique
    assert set(COLUMNS) <= set(df.columns)


@pytest.mark.parametrize("max_results", [5, None])
def test_failed_request_raises(
    monkeypatch: pytest.MonkeyPatch, max_results: int | None
) -> None:
    """A failed request raises, with one page of rows or paging with a cursor."""
    _answer(monkeypatch, None)

    with pytest.raises(SearchError, match="did not answer") as error:
        ArtFinder(print_status=False).search(query="nothing", max_results=max_results)

    assert error.value.url.startswith("https://api.crossref.org/works?")


def test_failed_funder_lookup_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A hit whose funder the Funder Registry does not answer for keeps the
    deposited name, and the search goes on."""
    _answer(monkeypatch, {"message": {"items": [_work()]}}, funder_body=None)

    with caplog.at_level(logging.WARNING, logger="artfinder.crossref"):
        df = ArtFinder(print_status=False).search(query="van der waals", max_results=5)

    assert len(df) == 1
    funder = next(
        funder for funder in df.iloc[0]["funders"] if funder.get("doi") == ROYAL_SOCIETY_DOI
    )
    assert funder["name"] == "Royal Society"
    assert "Funder Registry lookup failed" in caplog.text


def test_isearch_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The count is Crossref's total of results."""
    _answer(monkeypatch, {"message": {"total-results": 7}})

    assert ArtFinder(print_status=False).isearch(query="van der waals") == 7


def test_isearch_failed_request_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed count raises `SearchError`, not a `TypeError` on the missing total."""
    _answer(monkeypatch, None)

    with pytest.raises(SearchError, match="did not answer the count"):
        ArtFinder(print_status=False).isearch(query="van der waals")


def test_find_article_failed_title_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed title search gives an empty Series and says the search failed."""
    _answer(monkeypatch, None)

    with pytest.warns(UserWarning, match="failed"):
        result = ArtFinder(print_status=False).find_article(title="van der waals")

    assert result.empty


def test_search_error_crosses_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the real request thread, a failed search raises in the caller."""
    _stub_get(monkeypatch)

    error = _raised_within_deadline(
        lambda: ArtFinder(print_status=False).search(query="nothing", max_results=5)
    )

    assert isinstance(error, SearchError)


def test_coroutine_error_is_reraised(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception the request coroutine raises reaches the caller instead of
    leaving it waiting for a result."""
    _stub_get(monkeypatch, error=RuntimeError("boom"))

    error = _raised_within_deadline(
        lambda: AsyncHTTPRequest().get(
            url="https://api.crossref.org/works", print_progress=False
        )
    )

    assert isinstance(error, RuntimeError)
    assert str(error) == "boom"
