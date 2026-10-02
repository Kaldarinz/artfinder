"""
Tests for `ArtFinder.search` as a lookup of candidate articles by title.
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

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


def _answer(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any] | None) -> None:
    """
    Answer every Crossref request with one response body.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest's monkeypatch fixture.
    body : dict[str, Any] | None
        Response body, or None for a failed request.
    """

    def async_get(urls: list[str] | str, **kwargs: Any) -> dict[str, dict | None]:
        urls = [urls] if isinstance(urls, str) else urls
        return {url: copy.deepcopy(body) for url in urls}

    monkeypatch.setattr(AsyncHTTPRequest, "async_get", staticmethod(async_get))
    monkeypatch.setattr(Crossref, "_funder_registry", {})


def test_hits_have_candidate_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hit carries what a list of candidates shows."""
    with open(WORK_PATH, encoding="utf-8") as file:
        work = json.load(file)["message"]
    _answer(monkeypatch, {"message": {"items": [work]}})

    df = ArtFinder(print_status=False).search(query="van der waals", max_results=5)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["doi"] == "10.1021/acsnano.5c00546"
    assert row["title"] == "tunable nanostructuring for van der waals materials"
    assert row["journal"] == "ACS Nano"
    assert row["authors"][0]["lastname"] == "Tselikov"
    assert str(row["publication_date"]) == "2025-06-16"


@pytest.mark.parametrize("body", [{"message": {"items": []}}, None])
def test_no_hits_give_empty_frame(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any] | None
) -> None:
    """No hits, or a failed request, give an empty frame with each column once."""
    _answer(monkeypatch, body)

    df = ArtFinder(print_status=False).search(query="nothing", max_results=5)

    assert df.empty
    assert df.columns.is_unique
    assert set(COLUMNS) <= set(df.columns)
