"""
Classes for communication with the Crossref API.

This module is part of the Artfinder package.
Author: Anton Popov
email: a.popov.fizteh@gmail.com
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Self, TypeVar

import requests

from artfinder.dataclasses import CrossrefResource, CrossrefRateLimit

class Endpoint(ABC):

    ROW_LIMIT = 100
    "Maximum articles to be retrieved in a single request."

    def __init__(
        self,
        context: list[str] | None = None,
        request_params: dict[str, Any] | None = None,
        etiquette: Etiquette | None = None,
        timeout: int = 30,
        **kwargs
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
        self.timeout = timeout
        

    @property
    @abstractmethod
    def RESOURCE(self) -> CrossrefResource:
        """
        This property should be implemented in the child class.
        """
        pass

    @property
    def _update_rate_limits(self) -> CrossrefRateLimit:
        print(f"Updating rate limits")
        result = self.do_http_request(
            method="get",
            endpoint=self.request_endpoint,
            only_headers=True,
            custom_header=self.etiquette.header(),
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
            endpoint=self.request_endpoint,
            data=self.request_params,
            custom_header=self.etiquette.header(),
            timeout=self.timeout,
        ).json()

        return result.get("message_version", "undefined")

    def count(self) -> int:
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
        request_params["rows"] = 0

        result = self.do_http_request(
            method="get",
            endpoint=self.request_endpoint,
            data=request_params,
            custom_header=self.etiquette.header(),
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
            "get", self.request_endpoint, params=sorted_request_params
        ).prepare()

        return req.url

    @property
    def request_endpoint(self) -> str:
        """Request endpoint for http request."""
        return build_cr_endpoint(resource=self.RESOURCE, context=self.context)

    def __iter__(self):

        if any(value in self.request_params for value in ["sample", "rows"]):
            result = self.do_http_request(
                method="get",
                endpoint=self.request_endpoint,
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
                    endpoint=build_cr_endpoint(self.RESOURCE, self.context),
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

        return set(("etiquette", "request_params", "context", "rate_limits"))

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
        super().__init__(*args, **kwargs)

        self.rate_limits = kwargs.get('rate_limits') or self._update_rate_limits

    @property
    def RESOURCE(self) -> CrossrefResource:
        """works endpoint."""
        return CrossrefResource.WORKS

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
        This method can be chained with filter method.
        kwargs: CrossrefQueryField.
        """

        for field, value in kwargs.items():
            if field not in CrossrefQueryField:
                raise ValueError("Invalid query field name")
            self.request_params["query.%s" % field.replace("_", "-")] = value

        return self.from_self()

    def author(self, author: str) -> Self:
        """
        Search by author.
        """

        self.request_params["query." + CrossrefQueryField.AUTHOR] = author
        return self.from_self()

    def search(self, query: str) -> Self:
        """
        Bibliographic search.
        """
        self.request_params["query." + CrossrefQueryField.BIBLIOGRAPHIC] = query
        return self.from_self()

    def filter(self, **kwargs) -> Self:
        """
        This method can be chained with query.

        Valid filter fields are in CrossrefFilterField enum.
        """

        filter_validator = CrossrefFilterValidator()
        for field, value in kwargs.items():
            if isinstance(value, list):
                validated_values = [filter_validator(field, v) for v in value]
            else:
                validated_values = [filter_validator(field, value)]

            for i, v in enumerate(validated_values):
                if i == 0 and "filter" not in self.request_params:
                    self.request_params["filter"] = field + ":" + str(v)
                else:
                    self.request_params["filter"] += "," + field + ":" + str(v)

        return self.from_self()

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

        urls = [build_cr_endpoint(resource=self.RESOURCE, endpoint=doi) for doi in dois]
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


class CrossrefFilterValidator:
    """
    Validate filter values
    """

    VALIDATORS = {
        "alternative_id": "dummy",
        "archive": "archive",
        "article_number": "dummy",
        "assertion": "dummy",
        "assertion-group": "dummy",
        "award.funder": "dummy",
        "award.number": "dummy",
        "category-name": "dummy",
        "clinical-trial-number": "dummy",
        "container-title": "dummy",
        "content-domain": "dummy",
        "directory": "directory",
        "doi": "dummy",
        "from-accepted-date": "is_date",
        "from-created-date": "is_date",
        "from-deposit-date": "is_date",
        "from-event-end-date": "is_date",
        "from-event-start-date": "is_date",
        "from-index-date": "is_date",
        "from-issued-date": "is_date",
        "from-online-pub-date": "is_date",
        "from-posted-date": "is_date",
        "from-print-pub-date": "is_date",
        "from-pub-date": "is_date",
        "from-update-date": "is_date",
        "full-text.application": "dummy",
        "full-text.type": "dummy",
        "full-text.version": "dummy",
        "funder": "dummy",
        "funder-doi-asserted-by": "dummy",
        "group-title": "dummy",
        "has-abstract": "is_bool",
        "has-affiliation": "is_bool",
        "has-archive": "is_bool",
        "has-assertion": "is_bool",
        "has-authenticated-orcid": "is_bool",
        "has-award": "is_bool",
        "has-clinical-trial-number": "is_bool",
        "has-content-domain": "is_bool",
        "has-domain-restriction": "is_bool",
        "has-event": "is_bool",
        "has-full-text": "is_bool",
        "has-funder": "is_bool",
        "has-funder-doi": "is_bool",
        "has-license": "is_bool",
        "has-orcid": "is_bool",
        "has-references": "is_bool",
        "has-relation": "is_bool",
        "has-update": "is_bool",
        "has-update-policy": "is_bool",
        "is-update": "is_bool",
        "isbn": "dummy",
        "issn": "dummy",
        "license.delay": "is_integer",
        "license.url": "dummy",
        "license.version": "dummy",
        "location": "dummy",
        "member": "is_integer",
        "orcid": "dummy",
        "prefix": "dummy",
        "relation.object": "dummy",
        "relation.object-type": "dummy",
        "relation.type": "dummy",
        "type": "correct_doc_type",
        "type-name": "dummy",
        "until-accepted-date": "is_date",
        "until-created-date": "is_date",
        "until-deposit-date": "is_date",
        "until-event-end-date": "is_date",
        "until-event-start-date": "is_date",
        "until-index-date": "is_date",
        "until-issued-date": "is_date",
        "until-online-pub-date": "is_date",
        "until-posted-date": "is_date",
        "until-print-pub-date": "is_date",
        "until-pub-date": "is_date",
        "until-update-date": "is_date",
        "update-type": "dummy",
        "updates": "dummy",
    }

    def __call__(self, filter: str, value: str) -> Any:
        if value not in self.VALIDATORS:
            raise ValueError(f"Invalid filter {filter}. Valid filters: {self.VALIDATORS.keys()}")
        try:
            return getattr(self, self.VALIDATORS[filter])(value)
        except ValueError as exc:
            raise ValueError(f"Invalid value for filter {filter}: {exc.args[0]}") from exc

    @staticmethod
    def dummy(value: T) -> T:
        return value

    @staticmethod
    def is_date(value: str) -> datetime:
        """
        Validate date format.
        """
        try:
            return datetime.strptime(value, "%Y")
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m")
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d")
                except ValueError as exc:
                    raise ValueError(f"Invalid date {value}.")

    @staticmethod
    def is_integer(value: str|int) -> int:
        """
        Validate integer format.
        """
        try:
            value = int(value)
            if value >= 0:
                return value
            raise ValueError(f"Numerica value should be positive, but got: {value}")
        except ValueError:
            raise ValueError(f"Expected integer, but got: {value}")

    @staticmethod
    def is_bool(value: str | int) -> bool:

        true_vals = ["True", "true", "1"]
        false_vals = ["False", "false", "0"]

        if str(value) in true_vals:
            return True
        if str(value) in false_vals:
            return False
        raise ValueError(
            f"Expected boolean, but got: {value}. Expected values: {true_vals + false_vals}"
        )

    @staticmethod
    def correct_doc_type(value: str) -> str:
        """
        Validate document type.
        """
        if value in DocumentType:
            return value
        raise ValueError(
            f"Invalid document type {value}. Valid values are: {[doc.value for doc in DocumentType]}"
        )
    
    @staticmethod
    def archive(value: str) -> str:
        expected = ("Portico", "CLOCKSS", "DWT")

        if str(value) in expected:
            return value

        raise ValueError(f"Invalid archive {value}. Valid values are: {expected}")

    @staticmethod
    def directory(value: str) -> str:
        expected = "DOAJ"

        if str(value) in expected:
            return value

        msg = "Directory specified as {} but must be one of: {}".format(str(value), ", ".join(expected))
        raise ValueError(
            msg,
        )

class Etiquette:
    def __init__(
        self,
        application_name="undefined",
        application_url="undefined",
        contact_email="anonymous",
    ):
        self.application_name = application_name
        self.application_version = VERSION
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
