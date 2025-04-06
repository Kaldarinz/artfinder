"""API module."""

from __future__ import annotations

import warnings
import datetime
import random
import time
import threading
from queue import Queue
import sys
import logging
from typing import (
    Any,
    Dict,
    Generator,
    cast,
    Callable,
    ParamSpec,
    TypeVar,
    Coroutine,
    Literal,
    Self,
)
from abc import ABC, abstractmethod
from ast import literal_eval
import asyncio
import re

import aiohttp
from aiohttp import ClientResponse, ClientSession
import requests
from crossref.restful import (
    UrlSyntaxError,
    HTTPRequest,
)
from lxml import etree as xml
from typeguard import typechecked
import pandas as pd
from pandas import DataFrame

from artfinder.article import PubMedArticle, CrossrefArticle
from artfinder.dataclasses import *
from artfinder.helpers import (
    build_cr_endpoint,
    arrange_query,
    batches,
    get_range_date_from_query,
    get_range_months,
    get_range_years,
    get_search_term,
    LinePrinter,
    _execute_coro,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

# Base url for all queries
BASE_URL = "https://eutils.ncbi.nlm.nih.gov"

# Maximum retrieval records from PubMed api using esearch
MAX_RECORDS_PM = 9999

TIMEOUT = 10


class ArtFinder:
    """
    Base class for ArtFinder API.
    """

    def __init__(self, email: str | None = None) -> None:
        """
        Initialize the ArtFinder object.

        Returns
        -------
        None
        """
        self.cr = Crossref(email=email, app="artfinder")

    def find_article(
        self,
        doi: str | None = None,
        title: str | None = None,
        database: Literal["pubmed", "crossref", "all"] = "crossref",
    ) -> DataFrame:
        """
        Find an article by DOI or title.
        """

        if doi is None and title is None:
            raise ValueError("Either DOI or title must be provided.")
        if database in ["pubmed", "all"]:
            raise NotImplementedError("only crossref is supported for now")

        if title is not None:
            return self.cr.query(
                bibliographic=re.sub(r"\W+", "+", title.strip())
            ).get_df()


@typechecked
class Endpoint(ABC):

    ROW_LIMIT = 100

    def __init__(
        self,
        context: list[str] | None = None,
        request_params: dict[str, Any] | None = None,
        etiquette: Etiquette | None = None,
        rate_limits: CrossrefRateLimit | None = None,
        timeout: int = 30,
    ):
        self.do_http_request = HTTPRequest().do_http_request
        self.etiquette = etiquette or Etiquette()
        self.request_params = request_params or {}
        self.context = context
        """
        Context for the request. e.g. context=['types', 'journal-article'] and
        'works' as the endpoint will result in querying from 
        api.crossref.org/types/journal-article/works
        """
        self.rate_limits = rate_limits or self._update_rate_limits
        self.timeout = timeout


    @property
    @abstractmethod
    def ENDPOINT(self) -> CrossrefResource:
        """
        This property should be implemented in the child class.
        """
        pass

    @property
    def _update_rate_limits(self) -> CrossrefRateLimit:
        result = self.do_http_request(
            method="get",
            endpoint=self.request_url,
            only_headers=True,
            custom_header=self.custom_header,
            timeout=self.timeout,
        )
        return CrossrefRateLimit(
            limit=result.headers.get("x-rate-limit-limit", "undefined"),
            interval=result.headers.get("x-rate-limit-interval", "undefined"),
        )

    @property
    def request_params(self) -> dict[str, Any]:
        """
        This property retrieve the request parameters.
        """
        self._escaped_pagging()
        return self._request_params

    @request_params.setter
    def request_params(self, value: dict[str, Any]) -> None:
        """
        This property set the request parameters.
        """
        self._request_params = value

    def _escaped_pagging(self) -> None:
        """
        This method remove the offset and rows parameters from the
        request_params dictionary.
        This is used to build the url attribute.
        """

        escape_pagging = ["offset", "rows"]

        for item in escape_pagging:
            self._request_params.pop(item, None)

    @property
    def version(self) -> str:
        """
        This attribute retrieve the API version.
        """
        result = self.do_http_request(
            method="get",
            endpoint=self.request_url,
            data=self.request_params,
            custom_header=self.custom_header,
            timeout=self.timeout,
        ).json()

        return result.get("message_version", "undefined")

    def count(self):
        """
        This method retrieve the total of records resulting from a given query.

        This attribute can be used compounded with query, filter,
        sort, order and facet methods.

        Examples:
            >>> from crossref.restful import Works
            >>> Works().query('zika').count()
            3597
            >>> Works().query('zika').filter(prefix='10.1590').count()
            61
            >>> Works().query('zika').filter(prefix='10.1590').sort('published') \
                .order('desc').filter(has_abstract='true').count()
            14
            >>> Works().query('zika').filter(prefix='10.1590').sort('published') \
                .order('desc').filter(has_abstract='true').query(author='Marli').count()
            1
        """
        request_params = dict(self.request_params)
        request_url = str(self.request_url)
        request_params["rows"] = 0

        result = self.do_http_request(
            method="get",
            endpoint=request_url,
            data=request_params,
            custom_header=self.custom_header,
            timeout=self.timeout,
        ).json()

        return int(result["message"]["total-results"])

    @property
    def url(self):
        """
        This attribute retrieve the url that will be used as a HTTP request to
        the Crossref API.

        This attribute can be used compounded with query, filter,
        sort, order and facet methods.

        Examples:
            >>> from crossref.restful import Works
            >>> Works().query('zika').url
            'https://api.crossref.org/works?query=zika'
            >>> Works().query('zika').filter(prefix='10.1590').url
            'https://api.crossref.org/works?query=zika&filter=prefix%3A10.1590'
            >>> Works().query('zika').filter(prefix='10.1590').sort('published') \
                .order('desc').url
            'https://api.crossref.org/works?sort=published
            &order=desc&query=zika&filter=prefix%3A10.1590'
            >>> Works().query('zika').filter(prefix='10.1590').sort('published') \
                .order('desc').filter(has_abstract='true').query(author='Marli').url
            'https://api.crossref.org/works?sort=published
            &filter=prefix%3A10.1590%2Chas-abstract%3Atrue&query=zika&order=desc&query.author=Marli'
        """

        sorted_request_params = sorted([(k, v) for k, v in self.request_params.items()])
        req = requests.Request(
            "get", self.request_url, params=sorted_request_params
        ).prepare()

        return req.url

    def __iter__(self):

        if any(value in self.request_params for value in ["sample", "rows"]):
            result = self.do_http_request(
                method="get",
                endpoint=build_cr_endpoint(self.ENDPOINT, self.context),
                data=self.request_params,
                custom_header=self.etiquette.header(),
                timeout=self.timeout,
            )

            if result.status_code == 404:
                print("Not found any result.")
                return

            result = result.json()

            for item in result["message"]["items"]:
                yield item

            return

        else:
            request_params = dict(self.request_params)
            request_params["cursor"] = "*"
            request_params["rows"] = self.ROW_LIMIT
            while True:
                result = self.do_http_request(
                    method="get",
                    endpoint=build_cr_endpoint(self.ENDPOINT, self.context),
                    data=request_params,
                    custom_header=self.etiquette.header(),
                    timeout=self.timeout,
                )

                if result.status_code == 404:
                    print("Not found any result.")
                    return

                result = result.json()

                if len(result["message"]["items"]) == 0:
                    print("Empty result.")
                    return

                for item in result["message"]["items"]:
                    yield item

                request_params["cursor"] = result["message"]["next-cursor"]

    def init_params(self) -> set[str]:
        """Get list of parameters for initialization."""

        return set(
            ("etiquette", "request_params", "context", "rate_limits")
        )

    def from_self(self, **kwargs) -> Self:
        """
        Create a new instance of the class.
        """

        params_list = self.init_params()
        params = dict()
        for param in params_list:
            if param in kwargs:
                params[param] = kwargs[param]
            else:
                params[param] = getattr(self, param)
        return self.__class__(**params)


@typechecked
class Crossref(Endpoint):
    """Wrap around the Crossref API."""

    def __init__(self, email: str | None = None, *args, **kwargs) -> None:
        """
        Initialize the Crossref object.

        Parameters
        ----------
        email: str
            email of the user of the tool. This parameter
            is not required but kindly requested by Crossref.
        app: str
            name of the application that is executing the query.
            This parameter is not required but kindly requested by Crossref.

        Returns
        -------
        None
        """

        if kwargs.get("etiquette") is None:
            self.email = email or "anonymous"
            self.app = "artfinder"
            self.etiquette = Etiquette(
                application_name=self.app, contact_email=self.email
            )
            kwargs["etiquette"] = self.etiquette
        kwargs["resource"] = CrossrefResource.WORKS
        super().__init__(*args, **kwargs)

    def get_df(self) -> DataFrame:
        """
        Build data frame query results.
        """

        # Build list of CrossrefArticle objects
        all_articles = [CrossrefArticle(article) for article in self]
        # Create a DataFrame from the list of CrossrefArticle objects
        df = pd.concat([article.to_df() for article in all_articles], ignore_index=True)
        # If the DataFrame is empty, create an empty DataFrame with the correct columns
        if df.size == 0:
            df = DataFrame(columns=CrossrefArticle.get_all_slots())
        return df

    def query(self, **kwargs) -> Self:
        """
        This method can be used compounded with filter method.

        args: strings (String)

        kwargs: valid FIELDS_QUERY arguments.
        """

        for field, value in kwargs.items():
            if field not in CrossrefQueryField:
                msg = (
                    f"Field query {field!s} specified but there is no such field query for"
                    " this route."
                    f" Valid field queries for this route are: {', '.join(self.FIELDS_QUERY)}"
                )
                raise UrlSyntaxError(
                    msg,
                )
            request_params["query.%s" % field.replace("_", "-")] = value

        return self.__class__(
            request_url=request_url,
            request_params=request_params,
            context=context,
            etiquette=self.etiquette,
            timeout=self.timeout,
        )

    def filter(self, **kwargs):  # noqa: A003
        """
        This method retrieve an iterable object that implements the method
        __iter__. The arguments given will compose the parameters in the
        request url.

        This method can be used compounded and recursively with query, filter,
        order, sort and facet methods.

        kwargs: valid FILTER_VALIDATOR arguments. Replace `.` with `__` and
        `-` with `_` when using parameters.

        return: iterable object of Works metadata

        Example:
            >>> from crossref.restful import Works
            >>> works = Works()
            >>> query = works.filter(has_funder='true', has_license='true')
            >>> for item in query:
            ...     print(item['title'])
            ...
            ['Design of smiling-face-shaped band-notched UWB antenna']
            ['Phase I clinical and pharmacokinetic ... tients with advanced solid tumors']
            ...
        """
        context = str(self.context)
        request_url = build_url_endpoint(self.ENDPOINT, context)
        request_params = dict(self.request_params)

        for fltr, value in kwargs.items():
            decoded_fltr = fltr.replace("__", ".").replace("_", "-")
            if decoded_fltr not in self.FILTER_VALIDATOR.keys():
                msg = (
                    f"Filter {decoded_fltr!s} specified but there is no such filter for"
                    f" this route. Valid filters for this route"
                    f" are: {', '.join(self.FILTER_VALIDATOR.keys())}"
                )
                raise UrlSyntaxError(
                    msg,
                )

            if self.FILTER_VALIDATOR[decoded_fltr] is not None:
                if isinstance(value, list):
                    for v in value:
                        self.FILTER_VALIDATOR[decoded_fltr](str(v))
                else:
                    self.FILTER_VALIDATOR[decoded_fltr](str(value))
                    value = [value]

            for i, v in enumerate(value):
                if i == 0 and "filter" not in request_params:
                    request_params["filter"] = decoded_fltr + ":" + str(v)
                else:
                    request_params["filter"] += "," + decoded_fltr + ":" + str(v)

        return self.__class__(
            request_url=request_url,
            request_params=request_params,
            context=context,
            etiquette=self.etiquette,
            timeout=self.timeout,
        )

    async def _get_with_limit(
        self, dois: list[str], rate_limit: int = 10
    ) -> tuple[DataFrame, list[str]]:
        """
        Get a list of urls with a rate limit.

        Parameters
        ----------
        dois: list[str]
            List of articles doi to fetch.
        rate_limit: int
            Maximum number of requests per second.

        Returns
        -------
        DataFrame
            DataFrame with the results.
        list[str]
            List of doi that failed to fetch.
        """

        urls = [build_url_endpoint(endpoint=doi, context="works") for doi in dois]
        tot_urls = len(urls)
        failed_doi = []

        concur_limit = 5
        timeout_daley = 15
        concur_requests_limit = asyncio.Semaphore(concur_limit)
        timeout = asyncio.Event()
        timeout.set()
        last_fetch = time.time()
        printer = LinePrinter()

        async def fetch(session: ClientSession, url: str) -> dict:
            """
            Fetch a single article.
            """
            try:
                async with session.get(
                    url, headers={"User-Agent": str(self.etiquette)}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    # TODO: to handle 429 error properly, we should check current limits from headers
                    if response.status == 429:
                        logger.error(f"Rate limit exceeded for {url}")
                        # TODO: delay should be started from the laset recieved 429 error, not the first
                        if timeout.is_set():
                            timeout.clear()
                            await asyncio.sleep(delay=timeout_daley)
                            timeout.set()
                        return {}
                    else:
                        logger.error(f"Error fetching {url}: {response.status}")
                        return {}
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return {}

        async def fetch_with_limit(
            session: aiohttp.ClientSession, url: str, article_index: int
        ) -> CrossrefArticle | None:
            """
            Scheduled launch of article fetch with rate limit.
            """

            nonlocal last_fetch
            # respect concurrent requests limit
            async with concur_requests_limit:
                # wait for timeout if 429 error occured
                await timeout.wait()
                cur_time = time.time()
                if (cur_time - last_fetch) < (1 / rate_limit):
                    # wait until the next request can be made
                    await asyncio.sleep((1 / rate_limit) - (cur_time - last_fetch))
                # print progress in single line
                printer(f"{(article_index + 1)}/{tot_urls}: {url}")
                last_fetch = time.time()
                result = await fetch(session, url)
                if len(result):
                    try:
                        return CrossrefArticle(result["message"])
                    except Exception as e:
                        logger.error(f"Error parsing {url}: {e}")
                        failed_doi.append(dois[urls.index(url)])
                        return None

        async with ClientSession() as session:
            tasks = [fetch_with_limit(session, url, i) for i, url in enumerate(urls)]
            results = await asyncio.gather(*tasks)
            try:
                df = pd.concat(
                    (result.to_df() for result in results if result is not None)
                )
                printer(f"Obtained {len(df)} article{'s' if len(df) > 1 else ''}.")
                return df, failed_doi
            except ValueError:
                printer("No articles obtained.")
                return pd.DataFrame(), failed_doi
            finally:
                printer.close()

    def get_dois(
        self, dois: list[str], concurrent_lim: int = 50
    ) -> tuple[DataFrame, list[str]]:
        """
        Get all articles from a list of DOIs as dataframe.
        """

        return _execute_coro(self._get_with_limit, dois, rate_limit=concurrent_lim)

    def get_refs(
        self, df: DataFrame, concurrent_lim: int = 50
    ) -> tuple[DataFrame, list[str]]:
        """
        Get all references from articles in the DataFrame.
        """

        # Get all references from articles in the DataFrame
        all_refs = []
        for article in df["references"]:
            if article is not None:
                all_refs.extend(article)
        all_refs = list(set(all_refs))
        print(f"Found {len(all_refs)} unique references.")
        return _execute_coro(self._get_with_limit, all_refs, rate_limit=concurrent_lim)

class Etiquette:
    def __init__(
            self,
            application_name="undefined",
            application_version="undefined",
            application_url="undefined",
            contact_email="anonymous",
    ):
        self.application_name = application_name
        self.application_version = application_version
        self.application_url = application_url
        self.contact_email = contact_email

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

@typechecked
class PubMed:
    """Wrap around the PubMed API."""

    def __init__(
        self,
        tool: str = "my_tool",
        email: str = "my_email@example.com",
        api_key: str = "",
    ) -> None:
        """
        Initialize the PubMed object.

        Parameters
        ----------
        tool: String
            name of the tool that is executing the query.
            This parameter is not required but kindly requested by
            PMC (PubMed Central).
        email: String
            email of the user of the tool. This parameter
            is not required but kindly requested by PMC (PubMed Central).
        api_key: str
            the NCBI API KEY

        Returns
        -------
        None
        """
        # Store the input parameters
        self.tool = tool
        self.email = email

        # Keep track of the rate limit
        self._rateLimit: int = 3
        self._maxRetries: int = 10
        self._requestsMade: list[datetime.datetime] = []
        self.parameters: dict[str, str | int | list[str]]
        # Define the standard / default query parameters
        self.parameters = {"tool": tool, "email": email, "db": "pubmed"}

        if api_key:
            self.parameters["api_key"] = api_key
            self._rateLimit = 10

    def query(
        self,
        query: str,
        max_results: int = MAX_RECORDS_PM,
    ) -> Generator[PubMedArticle]:
        """
        Query the PubMed database.
        """

        # Get amount of articles that match the query
        total_articles = self.getTotalResultsCount(query)

        logger.info(f"Found: {total_articles} articles. Fetching...")
        # check if total articles is greater than MAX_RECORDS_PM
        # and check if the user requests more than MAX_RECORDS_PM
        if total_articles > MAX_RECORDS_PM and max_results > MAX_RECORDS_PM:
            article_ids = self._getArticleIdsMore10k(query=query)
        else:
            article_ids = self._getArticleIds(
                query=query,
                max_results=max_results,
            )

        # Get the articles themselves
        for batch in batches(article_ids, 250):
            yield from self._getArticles(article_ids=batch)

    def getArticles(self, article_ids: list[str]) -> Generator[PubMedArticle]:
        """
        Retrieve articles from PubMed.
        """
        for batch in batches(article_ids, 250):
            yield from self._getArticles(article_ids=batch)

    def getCitingArticles(
        self,
        pmid: str | int | None = None,
        doi: str | None = None,
        max_results: int = MAX_RECORDS_PM,
    ) -> Generator[PubMedArticle]:
        """
        Return the articles that cite the given article.
        """

        if pmid is None:
            raise NotImplementedError(
                "Citing articles search is only supported for pmid"
            )
        citing_article_ids = self._getCitingArticlesIDs(str(pmid), doi)
        total_citing_articles = len(citing_article_ids)
        logger.info(f"Found: {total_citing_articles} siting articles. Fetching...")
        # Check if the total number of citing articles is greater than MAX_RECORDS_PM
        # and check if the user requests more than MAX_RECORDS_PM
        if total_citing_articles > MAX_RECORDS_PM and max_results > MAX_RECORDS_PM:
            raise NotImplementedError(
                "Large citing articles count is not supported yet"
            )

        # Get the articles themselves
        for batch in batches(citing_article_ids, 250):
            yield from self._getArticles(article_ids=batch)

    def getCitingArticlesCount(self, pmid: str) -> int:
        """
        Return the number of articles that cite the given article.

        Parameters
        ----------
        pmid: String
            the PubMed ID of the article.

        Returns
        -------
        citing_articles_count: Int
            the number of articles that cite the given article.
        """

        return len(self._getCitingArticlesIDs(pmid))

    def getTotalResultsCount(self, query: str) -> int:
        """
        Return the total number of results for a query.
        """

        # Get the default parameters
        parameters = self.parameters.copy()

        # Add specific query parameters
        parameters["term"] = query
        # We are interested only in the amount of articles.
        # from https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
        parameters["rettype"] = "count"
        # Make the request (request a single article ID for this search)
        response = cast(
            Dict[str, Any],
            self._get(
                url="/entrez/eutils/esearch.fcgi",
                parameters=parameters,
                output="json",
            ),
        )

        # Get from the returned meta data the total number of available
        # results for the query
        total_results_count = int(response.get("esearchresult", {}).get("count", 0))

        # Return the total number of results (without retrieving them)
        return total_results_count

    def _exceededRateLimit(self) -> bool:
        """
        Check if we've exceeded the rate limit.

        Returns
        -------
        exceeded: Bool
            Whether or not the rate limit is exceeded.
        """
        # Remove requests from the list that are longer than 1 second ago
        self._requestsMade = [
            requestTime
            for requestTime in self._requestsMade
            if requestTime > datetime.datetime.now() - datetime.timedelta(seconds=1)
        ]

        # Return whether we've made more requests in the last second,
        # than the rate limit
        return len(self._requestsMade) > self._rateLimit

    def _wait_to_retry(self, attempt: int) -> None:
        """
        Calculate and wait the appropriate amount of time before a retry.

        Parameters.
        ----------
        attempt: int
            The current attempt number.
        """
        backoff_time = min(2**attempt, 32)  # Exponential backoff, capped at 32 seconds

        backoff_time += random.uniform(0, 1)  # Add jitter

        logger.debug(f"Waiting {backoff_time} seconds before retrying")
        time.sleep(backoff_time)

    def _get(
        self,
        url: str,
        parameters: dict[Any, Any] = {},
        output: str = "json",
    ) -> str | dict[str, Any]:
        """
        Make a request to PubMed.

        Parameters
        ----------
        url: str
            last part of the URL that is requested (will
            be combined with the base url)
        parameters: Dict
            parameters to use for the request
        output: Str
            type of output that is requested (defaults to
            JSON but can be used to retrieve XML)

        Returns
        -------
            - response      Dict / str, if the response is valid JSON it will
                            be parsed before returning, otherwise a string is
                            returend
        """

        if not parameters:
            parameters = self.parameters.copy()
        attempt = 0

        while self._exceededRateLimit():
            logger.debug("Rate limit exceeded, waiting before making new request")
            time.sleep(0.1)

        while attempt < self._maxRetries:
            try:
                logger.debug(f"Attempt {attempt + 1}/{self._maxRetries}")
                # xml or json
                parameters["retmode"] = output

                # Make the request to PubMed
                logger.debug(f"Requesting {BASE_URL}{url} with parameters {parameters}")
                response = requests.get(
                    f"{BASE_URL}{url}", timeout=TIMEOUT, params=parameters
                )
                # Check for any errors
                response.raise_for_status()
                logger.debug(f"Response status code: {response.status_code}")

                # Add this request to the list of requests made
                self._requestsMade.append(datetime.datetime.now())

                # Return the response
                if output == "json":
                    return cast(Dict[str, Any], response.json())
                else:
                    return response.text

            except Exception as exp:
                logger.debug(f"Error: {exp}")
                self._wait_to_retry(attempt)
                attempt += 1

        raise Exception(
            f"Failed to retrieve data from {BASE_URL}{url} "
            f"after {self._maxRetries} attempts"
        )

    def _getArticleIdsMonth(
        self,
        search_term: str,
        range_begin_date: datetime.date,
        range_end_date: datetime.date,
    ) -> list[str]:
        article_ids = []
        range_dates_month = get_range_months(range_begin_date, range_end_date)

        for begin_date, end_date in range_dates_month:
            arranged_query = arrange_query(
                search_term=search_term,
                start_date=begin_date,
                end_date=end_date,
            )
            article_ids += self._getArticleIds(
                query=arranged_query, max_results=MAX_RECORDS_PM
            )
        return article_ids

    def _getArticleIdsMore10k(self, query: str) -> list[str]:
        range_date = get_range_date_from_query(query)
        if range_date is None:
            raise Exception(
                f"Your query: {query} returns more than 9 999 results. "
                "PubMed database can only retrieve 9 999 records matching "
                "the query. "
                "Consider reducing the value of max_result to less than 9999"
                "or adding range date restriction to your query "
                " in the following format: \n"
                '(<your query>) AND ("YYYY/MM/DD"[Date - Publication]'
                ' : "YYYY/MM/DD"[Date - Publication])'
                ", so pymedx "
                "can split into smaller ranges to get more results."
            )

        search_term = get_search_term(query)

        range_begin_date, range_end_date = range_date

        date_ranges = get_range_years(range_begin_date, range_end_date)

        article_ids = []

        for begin_date, end_date in date_ranges:
            arranged_query = arrange_query(
                search_term=search_term,
                start_date=begin_date,
                end_date=end_date,
            )

            total_articles_year = self.getTotalResultsCount(arranged_query)

            if total_articles_year > MAX_RECORDS_PM:
                article_ids += self._getArticleIdsMonth(
                    search_term=search_term,
                    range_begin_date=begin_date,
                    range_end_date=end_date,
                )
            else:
                article_ids += self._getArticleIds(
                    query=arranged_query, max_results=MAX_RECORDS_PM
                )

        # Remove duplicated ids
        article_ids = list(set(article_ids))

        return article_ids

    def _getArticles(self, article_ids: list[str]) -> Generator[PubMedArticle]:
        """
        Retrieve articles from PubMed.
        """
        # Get the default parameters
        parameters = self.parameters.copy()
        parameters["id"] = article_ids

        # Make the request
        response = cast(
            str,
            self._get(
                url="/entrez/eutils/efetch.fcgi",
                parameters=parameters,
                output="xml",
            ),
        )

        # Parsing may fail if XML conatins encoding prolog.
        # We just ignore such responses and do not parse them.
        try:
            root = xml.fromstring(response)
        except ValueError:
            logger.warning("Failed to parse XML response")
            return

        unknown_article_counter = 0
        # Loop over the articles and construct article objects
        for article in root.iterchildren():
            if article.tag == "PubmedArticle":
                yield PubMedArticle(xml_element=article)
            else:
                unknown_article_counter += 1
        if unknown_article_counter > 0:
            logger.info(f"Unrecognized articles: {unknown_article_counter}")

    def _getArticleIds(
        self,
        query: str,
        max_results: int,
    ) -> list[str]:
        """Retrieve the article IDs for a query.

        Parameters
        ----------
        query: Str
            query to be executed against the PubMed database.
        max_results: Int
            the maximum number of results to retrieve.

        Returns
        -------
        article_ids: List
            article IDs as a list.
        """
        # Create a placeholder for the retrieved IDs
        article_ids = []

        # Get the default parameters
        parameters = self.parameters.copy()

        # Add specific query parameters
        parameters["term"] = query
        parameters["retmax"] = 500000
        parameters["datetype"] = "edat"

        retmax: int = cast(int, parameters["retmax"])

        # Calculate a cut off point based on the max_results parameter
        if max_results < retmax:
            parameters["retmax"] = max_results

        # Make the first request to PubMed
        response = cast(
            Dict[str, Any],
            self._get(
                url="/entrez/eutils/esearch.fcgi",
                parameters=parameters,
                output="json",
            ),
        )

        # Add the retrieved IDs to the list
        article_ids += response.get("esearchresult", {}).get("idlist", [])

        # Get information from the response
        total_result_count = int(response.get("esearchresult", {}).get("count", 0))
        retrieved_count = int(response.get("esearchresult", {}).get("retmax", 0))

        # If no max is provided (-1) we'll try to retrieve everything
        if max_results == -1:
            max_results = total_result_count

        # If not all articles are retrieved, continue to make requests until
        # we have everything
        while retrieved_count < total_result_count and retrieved_count < max_results:
            # Calculate a cut off point based on the max_results parameter
            if (max_results - retrieved_count) < cast(int, parameters["retmax"]):
                parameters["retmax"] = max_results - retrieved_count

            # Start the collection from the number of already retrieved
            # articles
            parameters["retstart"] = retrieved_count

            # Make a new request
            response = cast(
                Dict[str, Any],
                self._get(
                    url="/entrez/eutils/esearch.fcgi",
                    parameters=parameters,
                    output="json",
                ),
            )

            # Add the retrieved IDs to the list
            article_ids += response.get("esearchresult", {}).get("idlist", [])

            # Get information from the response
            retrieved_count += int(response.get("esearchresult", {}).get("retmax"))

        # Return the response
        return article_ids

    def _getCitingArticlesIDs(
        self, pmid: str | None = None, doi: str | None = None
    ) -> list[str]:
        """
        Return the IDs of the articles that cite the given article.
        """

        # Get the default parameters
        parameters = self.parameters.copy()

        # Remove db parameter
        parameters.pop("db")

        # Add specific query parameters
        parameters["dbfrom"] = "pubmed"
        parameters["linkname"] = "pubmed_pubmed_citedin"
        if pmid:
            parameters["id"] = pmid
        elif doi:
            parameters["doi"] = doi

        # Make the request
        response = cast(
            Dict[str, Any],
            self._get(
                url="/entrez/eutils/elink.fcgi",
                parameters=parameters,
                output="json",
            ),
        )

        # Get the number of citing articles
        citing_articles_ids = (
            response.get("linksets", [{}])[0]
            .get("linksetdbs", [{}])[0]
            .get("links", [])
        )

        # Return the number of citing articles
        return citing_articles_ids


def strict_filter(title: str) -> bool:

    # patterns
    parts = [
        r"(?=.*laser\w*)(?=.*\w*(gener|synth|prod|manufact|fabric)\w*)(?=.*(nano|colloid|quantum\sdot)\w*|.*\bnps\b)",
        r"(?=.*(nano|particle|cluster)\w*)(?=.*\b\w*(ablat|fragment)\w*)",
    ]
    pattern = r"(" + r"|".join(parts) + r")"
    # exclude patterns
    exlude_parts = [
        r"(?!.*nanostructur(ing|ed)\w*)",
    ]
    pattern += r"".join(exlude_parts)
    return re.search(pattern, title, re.IGNORECASE) is not None


def load_csv(path: str) -> DataFrame:
    """
    Load a CSV file into a DataFrame.
    """

    df = pd.read_csv(path)
    df = _format_df(df)
    return df


def _format_df(df: DataFrame) -> DataFrame:
    """
    Format the DataFrame to have the correct columns and types."
    """

    cols = list(set(CrossrefArticle.get_all_slots() + PubMedArticle.get_all_slots()))
    # Ensure all columns from cols are present in the DataFrame
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    # Convert to lower case
    for col in ["title", "abstract", "authors", "journal", "publisher"]:
        df[col] = df[col].str.lower()
    # convert to python objects to python types
    for col in ["license", "link", "authors", "references"]:
        df[col] = (
            df[col].fillna("None").str.replace("none", "None").transform(literal_eval)
        )
    # apply column types
    df = df.astype(CrossrefArticle.col_types())
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    return df
