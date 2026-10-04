"""
Tests for classifying the outcome of each file download.
"""

import asyncio
from pathlib import Path
from types import TracebackType
from typing import cast

import pytest
from aiohttp import ClientConnectionError, ClientSession, ConnectionTimeoutError

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


GOOD_URL = "https://example.org/content/good.pdf"
BAD_URL = "https://example.org/content/bad.pdf"


class FakeContent:
    """A response body served in one chunk."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    async def read(self, n: int) -> bytes:
        data, self.data = self.data, b""
        return data


class FakePDFResponse(FakeResponse):
    """A 200 response serving a PDF."""

    headers = {"Content-Type": "application/pdf"}

    def __init__(self, url: str, data: bytes) -> None:
        self.url = url
        self.content = FakeContent(data)


class FlakySession:
    """A session whose connection to `BAD_URL` times out."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def get(self, url: str) -> FakePDFResponse:
        if url == BAD_URL:
            raise self.error
        return FakePDFResponse(url, b"%PDF-1.4")

    async def __aenter__(self) -> "FlakySession":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@pytest.mark.parametrize(
    "error",
    [ConnectionTimeoutError("Connection timeout"), asyncio.TimeoutError(), ClientConnectionError()],
)
def test_connection_error_fails_one_file(
    error: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection error fails its own file and the rest of the batch still downloads."""
    monkeypatch.setattr(
        "artfinder.http_requests.ClientSession", lambda: FlakySession(error)
    )
    links: list[list[dict] | None] = [
        [{"content_type": "application/pdf", "url": url}] for url in (BAD_URL, GOOD_URL)
    ]
    paths = [str(tmp_path / "bad.pdf"), str(tmp_path / "good.pdf")]
    downloader = FileDownloader(links, paths, 1).download_files()
    assert downloader.failed == [(BAD_URL, error)]
    assert downloader.downloaded == [GOOD_URL]
    assert downloader.remaining_files_num == 0
    assert (tmp_path / "good.pdf").read_bytes() == b"%PDF-1.4"
    assert not (tmp_path / "bad.pdf").exists()
