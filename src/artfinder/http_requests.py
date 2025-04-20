# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
AsyncHTTP request class for making requests.

This module is part of the Artfinder package.
"""

import logging
from time import time
from typing import Any, Callable, Coroutine, TypeVar, ParamSpec, Iterator
from threading import Thread
from queue import Queue
import os

import asyncio
from aiohttp import (
    ClientSession,
    ClientTimeout,
    ClientError,
    ClientResponse,
)

from artfinder.dataclasses import CrossrefRateLimit
from artfinder.helpers import LinePrinter, MultiLinePrinter, PrinterLine

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class AsyncHTTPRequest:
    """
    Asynchronous HTTP request class for making get requests.
    """

    def __init__(
        self,
        email: str | None = None,
        rate_limit: CrossrefRateLimit = CrossrefRateLimit(50, 1),
        concurrency_limit: int = 5,
        concurrency_timeout: int = 15,
        max_retries: int = 5,
    ) -> None:
        """
        Initialize the AsyncHTTPRequest class.

        Parameters
        ----------
        etiquette : Etiquette
            The etiquette object to use for the requests.
        rate_limit : CrossrefRateLimit, optional
            The rate limit for the requests. Default is 50 requests per second.
        concurrency_limit : int, optional
            The maximum number of concurrent requests. Default is 5.
        concurrency_timeout : int, optional
            The timeout for concurrent requests in seconds. Default is 15 seconds.
        max_retries : int, optional
            The maximum number of retries for failed requests. Default is 5.
        """
        self.rate_limit = rate_limit
        self.concurrency_limit = concurrency_limit
        self.concurrency_timeout = concurrency_timeout
        self.etiquette = Etiquette(contact_email=email)
        self.max_retries = max_retries

    def _update_rate_limits(self, headers: dict[str, str]):

        try:
            rate_limit = int(headers.get("x-rate-limit-limit", 50))
        except:
            rate_limit = 50
        try:
            interval_value = int(headers.get("x-rate-limit-interval", "1s")[:-1])
        except:
            interval_value = 1

        interval_scope = headers.get("x-rate-limit-interval", "1s")[-1]

        if interval_scope == "m":
            interval_value = interval_value * 60

        if interval_scope == "h":
            interval_value = interval_value * 60 * 60

        self.rate_limit = CrossrefRateLimit(rate_limit, interval_value)

    async def _get(
        self,
        urls: list[str],
        params: dict[str, str] | None,
        only_headers: bool,
        timeout: int,
        print_progress: bool,
    ) -> dict[str, dict | None]:
        """
        Implementation of async HTTP GET request.

        Parameters
        ----------
        urls : list[str]
            List of URLs to request.
        params : dict[str, str] | str | None, optional
            Parameters to pass to the request. The same params will be passsed to all requests.
        only_headers : bool
            If True, only the headers will be returned.
        timeout : int
            The timeout for the request in seconds.
        print_progress : bool
            If True, print progress.

        Returns
        -------
        Dictionary, where keys are URLs and values are ClientResponse objects for all endpoints.
        """

        # Total number of URLs to process
        tot_urls = len(urls)

        # Additional timeout for rate-limiting
        rate_limit_extra_timeout = 0

        # Start time of the rate-limiting period
        rate_limited_start_time = 0

        # Semaphore to limit concurrent requests
        concur_requests_limit = asyncio.Semaphore(self.concurrency_limit)

        # Event to manage rate-limiting
        allowed_by_rate_limit = asyncio.Event()
        allowed_by_rate_limit.set()  # Initially allow requests

        # Track the time of the last request
        last_fetch_time = time()
        if print_progress:
            printer = LinePrinter()

        async def fetch(
            session: ClientSession,
            url: str,
        ) -> dict | None:
            """
            Fetch a single article.
            This function is called by fetch_with_limit().

            Parameters
            ----------
            session : ClientSession
                The aiohttp session to use for the request.
            url : str
                The URL to fetch.

            Returns
            -------
            dict | None
                The JSON response from the server or None if an error occurred.
            """

            left_retries = self.max_retries
            nonlocal rate_limit_extra_timeout, rate_limited_start_time
            try:
                while left_retries:
                    if only_headers:
                        method = session.head
                    else:
                        method = session.get
                    async with method(
                        url,
                        params=params,
                        headers=self.etiquette.header(),
                    ) as response:
                        if response.status == 200:
                            self._update_rate_limits(dict(response.headers))
                            if only_headers:
                                return dict(response.headers)
                            result = await response.json()
                            return result
                        # if the response is 429, wait for the rate limit and retry
                        elif response.status == 429:
                            # if allowed_by_rate_limit is set, then clear it for at
                            # least self.concurrency_timeout seconds.
                            if allowed_by_rate_limit.is_set():
                                rate_limited_start_time = time()
                                allowed_by_rate_limit.clear()
                                await asyncio.sleep(delay=self.concurrency_timeout)
                                # if while waiting for the rate limit, another 429
                                # errors occurred, then wait additionally for the timedelta between
                                # start of timelimiting and the latest 429 response
                                if rate_limit_extra_timeout:
                                    await asyncio.sleep(rate_limit_extra_timeout)
                                rate_limit_extra_timeout = 0
                                allowed_by_rate_limit.set()
                            # if allowed_by_rate_limit is not set, it means that we
                            # are already in the rate limit timeout, so set the timedelta
                            # between start of timelimiting and now to the rate_limit_extra_timeout
                            # if it is greater than the current rate_limit_extra_timeout
                            else:
                                if (
                                    additional_timeout := (
                                        time() - rate_limited_start_time
                                    )
                                    > rate_limit_extra_timeout
                                ):
                                    rate_limit_extra_timeout = additional_timeout
                        # retry for internal server errors
                        elif 500 <= response.status < 600:
                            logger.error(f"Server error {response.status} for {url}")
                        # return None for all other errors
                        else:
                            logger.error(f"Error fetching {url}: {response.status}")
                            return
                    left_retries -= 1
                logger.error(f"Max retries exceeded for {url}")
                return
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return

        async def fetch_with_limit(
            session: ClientSession, url: str, index: int
        ) -> tuple[str, dict | None]:
            """
            Scheduled launch of article fetch with rate limit.

            Parameters
            ----------
            session : ClientSession
                The aiohttp session to use for the request.
            url : str
                The URL to fetch.
            index : int
                The index of the request in the list of URLs.

            Returns
            -------
            dict | None
                The JSON response from the server or None if an error occurred.
            """

            nonlocal last_fetch_time
            # respect concurrent requests limit
            async with concur_requests_limit:
                # wait for timeout if 429 error occured
                await allowed_by_rate_limit.wait()
                cur_time = time()
                # wait until the next request can be made
                if (
                    delay := self.rate_limit.interval / self.rate_limit.limit
                    - (cur_time - last_fetch_time)
                ) > 0:
                    await asyncio.sleep(delay)
                if print_progress:
                    printer(f"Fetching {(index + 1)}/{tot_urls}: {url}")
                last_fetch_time = time()
                return url, await fetch(session, url)

        async with ClientSession(timeout=ClientTimeout(total=timeout)) as session:
            tasks = [fetch_with_limit(session, url, i) for i, url in enumerate(urls)]
            gatered = await asyncio.gather(*tasks)

        results = {result[0]: result[1] for result in gatered}
        if print_progress:
            printer(f"Fetched {len(results)}/{tot_urls} URLs.")
        return results

    def async_get(
        self,
        urls: list[str] | str,
        params: dict[str, str] | None = None,
        only_headers: bool = False,
        timeout: int = 60,
        print_progress: bool = True,
    ) -> dict[str, dict | None]:
        """
        Make async HTTP request.

        Parameters
        ----------
        urls : list[str]
            List of URLs to request.
        params : dict[str, str] | str | None, optional
            Parameters to pass to the request. The same params will be passsed to all requests.
        only_headers : bool, optional
            If True, only the headers will be returned.
        timeout : int, optional
            The timeout for the request in seconds.
        print_progress : bool, optional
            If True, print progress.

        Returns
        -------
        Dictionary, where keys are URLs and values are dicts with response bodies.
        """
        if isinstance(urls, str):
            urls = [urls]
        return _execute_coro(
            self._get,
            urls=urls,
            params=params,
            only_headers=only_headers,
            timeout=timeout,
            print_progress=print_progress,
        )

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        only_headers: bool = False,
        timeout: int = 60,
        print_progress: bool = True,
    ) -> dict:
        """
        Make HTTP request.

        Parameters
        ----------
        url : str
            URLs to request.
        params : dict[str, str] | str | None, optional
            Parameters to pass to the request. The same params will be passsed to all requests.
        only_headers : bool, optional
            If True, only the headers will be returned.
        timeout : int, optional
            The timeout for the request in seconds.
        print_progress : bool, optional
            If True, print progress.

        Returns
        -------
        Response body.
        """
        return (
            self.async_get(
                urls=url,
                params=params,
                only_headers=only_headers,
                timeout=timeout,
                print_progress=print_progress,
            ).get(url, {})
            or {}
        )


class FileDownloader:
    """
    Class to handle file downloading.
    """

    def __init__(
        self,
        links: list[list[dict] | None],
        save_paths: list[str],
        concurency_limit: int,
    ) -> None:
        """
        Create a FileDownloader instance.

        Parameters
        ----------
        links : list[list[dict] | None]
            "link" field from Article object.
        save_paths : list[str]
            Paths to save the downloaded files.
        concurency_limit : int
            Maximum number of concurrent downloads.
        """
        urls = [self.get_pdf_link(link) for link in links]
        pairs = [(url, path) for url, path in zip(urls, save_paths) if url is not None]
        self.urls = [pair[0] for pair in pairs]
        self.save_paths = [pair[1] for pair in pairs]
        self.downloaded = []
        self.restricted = []
        self.missing = []
        self.failed = []
        self.concurrency_limiter = asyncio.Semaphore(concurency_limit)
        self.chunk_size = 1024
        self.printer = MultiLinePrinter(concurency_limit + 1)
        self.status_line = self.printer.get_line()
        self.status_line(f"{self.total_urls_num} files to download.")

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(zip(self.urls, self.save_paths))

    @property
    def processed_files_num(self) -> int:
        return (
            len(self.downloaded)
            + len(self.restricted)
            + len(self.missing)
            + len(self.failed)
        )

    @property
    def total_urls_num(self) -> int:
        return len(self.urls)

    @property
    def remaining_files_num(self) -> int:
        return self.total_urls_num - self.processed_files_num

    def print_status(self) -> None:
        self.status_line.update(
            f"{self.total_urls_num} links. {len(self.downloaded)} downloaded. "
            + f"{len(self.restricted)} restricted. {len(self.missing)} missing. "
            + f"{len(self.failed)} failed. {self.remaining_files_num} remaining."
        )
        self.printer.print()

    @staticmethod
    def get_pdf_link(link_entry: list[dict] | None) -> str | None:
        """
        Extracts the PDF link from the provided link entry.
        """

        if link_entry is None or len(link_entry) == 0:
            return None
        for entry in link_entry:
            if entry.get("content-type") == "application/pdf":
                return entry.get("url")

    async def _download_files(self) -> "FileDownloader":
        """
        Download files from the provided URLs and save them to the specified paths.

        This method should not be called directly. Instead, use the `full_texts_from_urls` function or similar.

        This method handles downloading files asynchronously with a specified level of concurrency.
        It tracks the status of downloads, including successful downloads, restricted access,
        missing files, and failed downloads.

        Returns
        -------
        FileDownloader
            The instance of the FileDownloader class with updated download status.

        Notes
        -----
        - This method uses aiohttp for asynchronous HTTP requests.
        - It provides real-time progress updates for each file being downloaded.
        - Handles HTTP status codes to categorize downloads into different statuses:
          - 200: Successful download.
          - 403: Restricted access.
          - 404: File not found.
          - Other: Failed download with the corresponding HTTP status code.
        - Progress is displayed using a MultiLinePrinter for better visualization.

        Example
        -------
        >>> downloader = FileDownloader(urls, save_paths, 5)
        >>> await downloader._download_files(urls, save_paths, 5)
        """

        self.status_line.update(f"Downloading {self.total_urls_num} files...")
        printer_task = asyncio.create_task(self.periodic_print(0.2))
        async with ClientSession() as session:
            tasks = [
                self.download_file(session, url, save_path) for url, save_path in self
            ]
            await asyncio.gather(*tasks)
        printer_task.cancel()
        try:
            await printer_task
        except asyncio.CancelledError:
            pass
        self.printer.close()
        return self

    async def periodic_print(self, period: float) -> None:
        """
        Periodically print the download status.
        """
        try:
            while True:
                self.print_status()
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            # By some reason, if we call this, in VScode all output will be lost
            # self.printer.print()
            pass

    async def download_file(
        self, session: ClientSession, url: str, save_path: str
    ) -> None:
        """
        Download a single file and update its status.

        Parameters
        ----------
        session : ClientSession
            The aiohttp session used for making HTTP requests.
        url : str
            The URL of the file to download.
        save_path : str
            The path where the downloaded file will be saved.
        """

        filename = os.path.basename(save_path)
        async with self.concurrency_limiter:
            with self.printer.get_line() as progress_line:
                progress_line(f"Downloading File: {filename}")
                async with session.get(url) as response:
                    if response.status == 200:
                        # Check if the response is a CAPTCHA
                        content_type = response.headers.get("Content-Type", "")
                        if "text/html" in content_type:
                            content = await response.text()
                            if "captcha" in content.lower() or "verify" in content.lower():
                                progress_line.update(
                                    f"CAPTCHA detected. File: {filename}. URL: {url}"
                                )
                                self.failed.append((url, "CAPTCHA detected"))
                        else:
                            await self._write_file(save_path, response, progress_line)
                    elif response.status == 403:
                        self.restricted.append(url)
                        progress_line.update(
                            f"Access denied. HTTP status: {response.status}. File: {filename}"
                        )
                    elif response.status == 404:
                        self.missing.append(url)
                        progress_line.update(
                            f"File not found. HTTP status: {response.status}. File: {filename}"
                        )
                    else:
                        self.failed.append((url, response.status))
                        progress_line.update(
                            f"Failed to download file. HTTP status: {response.status}. File: {filename}"
                        )
                self.print_status()

    async def _write_file(
        self, path: str, response: ClientResponse, progress_line: PrinterLine
    ) -> bool:
        """
        Write the response content to a file and update the progress.
        This method implies that the response status is 200.
        If an error occurs, the partially downloaded file will be deleted.

        Parameters
        ----------
        path : str
            The path where the file will be saved.
        response : ClientResponse
            The response object containing the file content.
        progress_line : PrinterLine
            The line printer object used for displaying progress.
        
        Returns
        -------
        bool
            True if the file was downloaded successfully, False otherwise.
        """

        filename = os.path.basename(path)
        # Get total file size in kb
        total_size = int(
            response.headers.get("Content-Length", 0)
        ) / 1024  
        downloaded_size = 0
        with open(path, "wb") as f:
            try:
                while chunk := await response.content.read(self.chunk_size):
                    f.write(chunk)
                    downloaded_size += len(chunk) / 1024
                    # Update progress for this task
                    progress = (downloaded_size / total_size) * 100 if total_size else 0
                    # Format progress message
                    if total_size:
                        progress_line.update(
                            f"Downloading: {progress:.2f}% ({int(downloaded_size)}/{int(total_size)})"
                          + f" kb. File: {filename})"
                        )
                    else:
                        progress_line.update(
                            f"Downloading: {int(downloaded_size)} kb. File: {filename}"
                        )
            except (ClientError, asyncio.IncompleteReadError) as e:
                progress_line.update(f"Error: {e}. File: {filename}")
                self.failed.append((response.url, e))
                # Delete the partially downloaded file
                if os.path.exists(path):
                    os.remove(path)
                return False
        self.downloaded.append(response.url)
        progress_line.update(f"Downloaded {int(downloaded_size)} kb. File: {filename}")
        return True

    def download_files(self) -> "FileDownloader":
        """
        Download files using URLs and paths provided at initialization.
        """
        return _execute_coro(self._download_files)


class Etiquette:
    def __init__(
        self,
        application_name: str = "undefined",
        application_url: str = "undefined",
        contact_email: str | None = None,
    ):
        self.application_name = application_name
        # TODO: propper version import
        self.application_version = "0.1.0"
        self.application_url = application_url
        self.contact_email = contact_email or "anon"

    def __str__(self):
        return "{}/{} ({}; mailto:{})".format(
            self.application_name,
            self.application_version,
            self.application_url,
            self.contact_email,
        )

    def header(self) -> dict[str, str]:
        """
        This method returns the etiquette header.
        """

        return {"user-agent": str(self)}


def _execute_coro(func: Callable[P, Coroutine[Any, Any, T]], *args, **kwargs) -> T:
    """
    Launch function asyncronously in separate thread.
    """

    result_queue = Queue()

    def get_func():
        result = asyncio.run(func(*args, **kwargs))
        result_queue.put(result)

    thread = Thread(target=get_func)
    thread.start()
    thread.join()
    return result_queue.get()
