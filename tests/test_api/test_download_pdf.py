"""
Tests for picking the PDF link of each article before downloading it.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from artfinder.api import ArtFinder
from artfinder.article import ArticleCollection, CrossrefArticle, load_csv
from artfinder.http_requests import FileDownloader

PDF_URL = "https://example.org/content/Article.PDF"

RECORD: dict[str, Any] = {
    "DOI": "10.1000/example",
    "title": ["An example"],
    "container-title": ["Example Journal"],
    "is-referenced-by-count": 0,
    "link": [
        {"URL": "https://example.org/content/article.xml", "content-type": "text/xml"},
        {"URL": PDF_URL, "content-type": "application/pdf"},
    ],
}


@pytest.fixture
def no_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make `FileDownloader.download_files` return without touching the network.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest's monkeypatch fixture.
    """

    monkeypatch.setattr(FileDownloader, "download_files", lambda self: self)


def _download(articles: CrossrefArticle | pd.DataFrame, tmp_path: Path) -> FileDownloader:
    """
    Run `ArtFinder.download_pdf` up to the point of downloading.

    Parameters
    ----------
    articles : CrossrefArticle | pd.DataFrame
        Articles to download.
    tmp_path : Path
        Directory to save the files to.

    Returns
    -------
    FileDownloader
        The downloader, holding the URLs it would fetch.
    """

    return ArtFinder(print_status=False).download_pdf(
        articles, path=str(tmp_path), name="doi"
    )


@pytest.mark.usefixtures("no_download")
class TestPdfLink:
    """The PDF link is picked from the article's `links` column."""

    def test_from_article(self, tmp_path: Path) -> None:
        """An article converted in memory yields its PDF link, case intact."""
        downloader = _download(CrossrefArticle(dict(RECORD)), tmp_path)
        assert downloader.urls == [PDF_URL]
        assert downloader.save_paths == [str(tmp_path / "10.1000_example.pdf")]

    def test_from_csv(self, tmp_path: Path) -> None:
        """An article read back from a CSV file yields the same link."""
        path = tmp_path / "articles.csv"
        ArticleCollection([dict(RECORD)]).to_df().to_csv(path, index=False)
        downloader = _download(load_csv(str(path)), tmp_path)
        assert downloader.urls == [PDF_URL]

    def test_article_without_pdf_is_skipped(self, tmp_path: Path) -> None:
        """An article with no PDF link, or no links at all, is not downloaded."""
        no_pdf = dict(RECORD, link=[RECORD["link"][0]])
        no_links = {key: val for key, val in RECORD.items() if key != "link"}
        articles = ArticleCollection([no_pdf, no_links, dict(RECORD)]).to_df()
        assert _download(articles, tmp_path).urls == [PDF_URL]
