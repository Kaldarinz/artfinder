"""
Classes for communication with the Pubmed API.

This module is part of the Artfinder package.
Author: Anton Popov
email: a.popov.fizteh@gmail.com
"""
from datetime import datetime, timedelta, date
import random
import time
import logging
from typing import Any, Dict, Generator, cast

from typeguard import typechecked
import requests
from lxml import etree as xml

from artfinder.article import PubMedArticle
from artfinder.helpers import (
    arrange_query,
    batches,
    get_range_date_from_query,
    get_range_months,
    get_range_years,
    get_search_term,
)

logger = logging.getLogger(__name__)

# Base url for all queries
BASE_URL = "https://eutils.ncbi.nlm.nih.gov"

# Maximum retrieval records from PubMed api using esearch
MAX_RECORDS_PM = 9999

TIMEOUT = 10

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
        self._requestsMade: list[datetime] = []
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
            if requestTime > datetime.now() - timedelta(seconds=1)
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
                self._requestsMade.append(datetime.now())

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
        range_begin_date: date,
        range_end_date: date,
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
