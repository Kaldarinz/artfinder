"""
Tests for naming a work's funders from the Crossref Funder Registry.
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from artfinder.article import CrossrefArticle
from artfinder.crossref import Crossref
from artfinder.crossref_helpers import funder_registry_id
from artfinder.http_requests import AsyncHTTPRequest

DATA = Path(__file__).parent / "data"

WORK_DOI = "10.1021/acsnano.5c00546"
WORK_URL = f"https://api.crossref.org/works/{WORK_DOI}"
# The work deposits this funder's name with every non-ASCII character as "?".
MEYS_ID = "501100001823"
MEYS_URL = f"https://api.crossref.org/funders/{MEYS_ID}"
MEYS_DEPOSITED = "Ministerstvo ?kolstv?, Ml?de?e a Telov?chovy"
MEYS_NAME = "Ministerstvo Školství, Mládeže a Tělovýchovy"


def _load(name: str) -> dict[str, Any]:
    """
    Read a saved Crossref API response.

    Parameters
    ----------
    name : str
        File name under `data/`.

    Returns
    -------
    dict[str, Any]
        The response body.
    """

    with open(DATA / name, encoding="utf-8") as file:
        return json.load(file)


WORK: dict[str, Any] = _load("work_10.1021_acsnano.5c00546.json")
MEYS: dict[str, Any] = _load(f"funder_{MEYS_ID}.json")


class FakeCrossref:
    """Stand-in for the Crossref API, answering from a table of URLs."""

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        """
        Parameters
        ----------
        responses : dict[str, dict[str, Any]]
            Response body by URL. Any other URL fails, as a 404 does.
        """

        self.responses = responses
        self.queued: dict[str, list[dict[str, Any]]] = {}
        "Answers by URL, one per request, taking precedence over `responses`."
        self.requested: list[str] = []

    def async_get(
        self,
        urls: list[str] | str,
        params: dict[str, str] | None = None,
        only_headers: bool = False,
        timeout: int = 60,
        print_progress: bool = True,
    ) -> dict[str, dict | None]:
        """
        Answer the way `AsyncHTTPRequest.async_get` does.

        Parameters
        ----------
        urls : list[str] | str
            URLs to request.
        params : dict[str, str] | None, optional
            Query parameters, ignored.
        only_headers : bool, optional
            Ignored.
        timeout : int, optional
            Ignored.
        print_progress : bool, optional
            Ignored.

        Returns
        -------
        dict[str, dict | None]
            Response body by URL, None for a failed request.
        """

        urls = [urls] if isinstance(urls, str) else urls
        self.requested.extend(urls)
        return {url: copy.deepcopy(self._answer(url)) for url in urls}

    def _answer(self, url: str) -> dict[str, Any] | None:
        """
        Answer one request.

        Parameters
        ----------
        url : str
            Requested URL.

        Returns
        -------
        dict[str, Any] | None
            The next queued answer for the URL, or its fixed response; None
            once its queue runs out, or when there is neither.
        """

        if url in self.queued:
            queue = self.queued[url]
            return queue.pop(0) if queue else None
        return self.responses.get(url)


@pytest.fixture
def crossref(monkeypatch: pytest.MonkeyPatch) -> FakeCrossref:
    """
    Route every Crossref request to a `FakeCrossref`, with an empty funder cache.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest's monkeypatch fixture.

    Returns
    -------
    FakeCrossref
        The fake, knowing the work and its Ministry funder only.
    """

    fake = FakeCrossref({WORK_URL: WORK, MEYS_URL: MEYS})
    monkeypatch.setattr(AsyncHTTPRequest, "async_get", staticmethod(fake.async_get))
    monkeypatch.setattr(Crossref, "_funder_registry", {})
    return fake


def _funders() -> list[dict[str, Any]]:
    """
    Fetch the work and return its funders.

    Returns
    -------
    list[dict[str, Any]]
        The `funders` column of the work's row.
    """

    return Crossref(print_status=False).doi(WORK_DOI).iloc[0]["funders"]


def _by_doi(funders: list[dict[str, Any]], funder_id: str) -> dict[str, Any]:
    """
    Pick one funder by its Funder Registry id.

    Parameters
    ----------
    funders : list[dict[str, Any]]
        Funder records.
    funder_id : str
        Funder Registry id.

    Returns
    -------
    dict[str, Any]
        The funder record.
    """

    return next(f for f in funders if f.get("doi") == f"10.13039/{funder_id}")


@pytest.mark.parametrize(
    ("doi", "expected"),
    [
        ("10.13039/501100001823", MEYS_ID),
        ("https://doi.org/10.13039/501100001823", MEYS_ID),
        ("http://dx.doi.org/10.13039/100000001", "100000001"),
        ("10.1021/acsnano.5c00546", None),
        (None, None),
    ],
)
def test_funder_registry_id(doi: str | None, expected: str | None) -> None:
    """The id is the suffix of a Funder Registry DOI, bare or as a URL."""
    assert funder_registry_id(doi) == expected


@pytest.mark.usefixtures("crossref")
class TestRegistryName:
    """A funder with a Funder Registry DOI is named as the registry names it."""

    def test_deposit_is_corrupted(self) -> None:
        """The fixture carries the name as the publisher deposited it."""
        names = [funder["name"] for funder in WORK["message"]["funder"]]
        assert MEYS_DEPOSITED in names

    def test_name_comes_from_registry(self) -> None:
        """The corrupted name is replaced by the canonical one."""
        assert _by_doi(_funders(), MEYS_ID)["name"] == MEYS_NAME

    def test_alt_names_come_from_registry(self) -> None:
        """The registry's alternative names are listed, in its order."""
        assert (
            _by_doi(_funders(), MEYS_ID)["alt_names"]
            == MEYS["message"]["alt-names"]
        )

    def test_doi_and_award_are_kept(self) -> None:
        """Only the names change; the DOI and award number are as deposited."""
        funder = _by_doi(_funders(), MEYS_ID)
        assert funder["doi"] == f"10.13039/{MEYS_ID}"
        assert funder["number"] == "LL2101"


@pytest.mark.usefixtures("crossref")
class TestDepositedName:
    """A funder the registry cannot name keeps its deposited name."""

    def test_funder_without_doi(self) -> None:
        """A funder with no DOI is not looked up."""
        funder = next(f for f in _funders() if "doi" not in f)
        assert funder == {
            "name": "French government",
            "number": "A*MIDEX AMX-22-RE-AB-107",
        }

    def test_failed_lookup(self) -> None:
        """A funder whose registry lookup fails keeps its name, with no alt-names."""
        funder = _by_doi(_funders(), "501100000288")
        assert funder["name"] == "Royal Society"
        assert "alt_names" not in funder


class TestCache:
    """Registry lookups are cached per funder id."""

    def test_one_lookup_per_funder(self, crossref: FakeCrossref) -> None:
        """A funder named by several works, or in several queries, is fetched once."""
        other_doi = "10.1000/other"
        other = copy.deepcopy(WORK)
        other["message"]["DOI"] = other_doi
        crossref.responses[f"https://api.crossref.org/works/{other_doi}"] = other

        df = Crossref(print_status=False).get_dois([WORK_DOI, other_doi])
        Crossref(print_status=False).doi(WORK_DOI)

        assert crossref.requested.count(MEYS_URL) == 1
        assert [_by_doi(funders, MEYS_ID)["name"] for funders in df["funders"]] == [
            MEYS_NAME,
            MEYS_NAME,
        ]

    def test_failed_lookup_is_retried(self, crossref: FakeCrossref) -> None:
        """A failed lookup is not cached, so a later query tries again."""
        royal_society = "https://api.crossref.org/funders/501100000288"
        Crossref(print_status=False).doi(WORK_DOI)
        Crossref(print_status=False).doi(WORK_DOI)
        assert crossref.requested.count(royal_society) == 2


def test_query_results_are_named_from_registry(crossref: FakeCrossref) -> None:
    """Works found by a query are named the same way as works fetched by DOI."""
    search = Crossref(print_status=False).search("laser ablation")
    crossref.responses[search.request_url] = {"message": {"items": [WORK["message"]]}}
    df = search.get_df(max_results=1)
    assert _by_doi(df.iloc[0]["funders"], MEYS_ID)["name"] == MEYS_NAME


def test_failed_doi_fetch_gives_empty_frame(crossref: FakeCrossref) -> None:
    """A DOI the API does not answer for gives an empty frame, not a KeyError."""
    df = Crossref(print_status=False).doi("10.1000/missing")
    assert df.empty
    assert list(df.columns) == CrossrefArticle.get_all_slots()


@pytest.mark.parametrize("max_results", [1, None])
def test_failed_query_gives_empty_frame(
    crossref: FakeCrossref, max_results: int | None
) -> None:
    """A query the API does not answer gives an empty frame, with or without rows."""
    df = Crossref(print_status=False).search("laser ablation").get_df(max_results)
    assert df.empty
    assert list(df.columns) == CrossrefArticle.get_all_slots()


def test_failed_page_keeps_earlier_pages(crossref: FakeCrossref) -> None:
    """A page that fails mid-pagination ends the results after the pages before it."""
    search = Crossref(print_status=False).search("laser ablation")
    crossref.queued[search.request_url] = [
        {"message": {"items": [WORK["message"]], "next-cursor": "c2"}}
    ]
    df = search.get_df()
    assert list(df["doi"]) == [WORK_DOI.lower()]
