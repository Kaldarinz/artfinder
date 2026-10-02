"""
Tests for classifying the response to a single file download.
"""

import asyncio
from pathlib import Path
from types import TracebackType
from typing import cast

import pytest
from aiohttp import ClientSession

from artfinder.http_requests import FileDownloader

PDF_URL = "https://example.org/content/article.pdf"


class FakeResponse:
    """A 200 response serving an HTML page."""

    status = 200
    headers = {"Content-Type": "text/html; charset=utf-8"}

    def __init__(self, body: str) -> None:
        self.body = body

    async def text(self) -> str:
        return self.body

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class FakeSession:
    """A session answering every request with the same HTML page."""

    def __init__(self, body: str) -> None:
        self.body = body

    def get(self, url: str) -> FakeResponse:
        return FakeResponse(self.body)


async def _download_html(body: str, tmp_path: Path) -> FileDownloader:
    """
    Download one file whose URL serves an HTML page.

    Parameters
    ----------
    body : str
        Body of the HTML page.
    tmp_path : Path
        Directory to save the file to.

    Returns
    -------
    FileDownloader
        The downloader after the download.
    """

    links: list[list[dict] | None] = [[{"content_type": "application/pdf", "url": PDF_URL}]]
    downloader = FileDownloader(links, [str(tmp_path / "article.pdf")], 1)
    session = cast(ClientSession, FakeSession(body))
    await downloader.download_file(session, PDF_URL, downloader.save_paths[0])
    return downloader


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("<html>Please complete the CAPTCHA</html>", "CAPTCHA detected"),
        ("<html>Example Journal: An example</html>", "HTML page instead of PDF"),
    ],
)
def test_html_page_is_failed(body: str, reason: str, tmp_path: Path) -> None:
    """An HTML page served for a PDF counts as failed, CAPTCHA or not."""
    downloader = asyncio.run(_download_html(body, tmp_path))
    assert downloader.failed == [(PDF_URL, reason)]
    assert downloader.remaining_files_num == 0
    assert not (tmp_path / "article.pdf").exists()
