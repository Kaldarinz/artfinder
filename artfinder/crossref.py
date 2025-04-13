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
import logging

import requests
import pandas as pd
from pandas import DataFrame

from artfinder.dataclasses import CrossrefResource, CrossrefQueryField, DocumentType
from artfinder.http_requests import AsyncHTTPRequest
from artfinder.crossref_helpers import build_cr_endpoint
from artfinder.article import CrossrefArticle

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Endpoint(ABC):

    ROW_LIMIT = 100
    "Maximum articles to be retrieved in a single request."

    def __init__(
        self,
        context: list[str] | None = None,
        request_params: dict[str, Any] | None = None,
        email: str | None = None,
        **kwargs,
    ):
        self.do_http_request = AsyncHTTPRequest(email).async_get
        self.request_params = request_params or {}
        self.context = context
        self.email = email
        """
        Context for the request. e.g. context=['types', 'journal-article'] and
        RESOURCE='works' will result in querying from 
        api.crossref.org/types/journal-article/works
        """

    @property
    @abstractmethod
    def RESOURCE(self) -> CrossrefResource:
        """
        This property should be implemented in the child class.
        """
        pass

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
        This method removes the offset and rows parameters from the
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
        result = (
            self.do_http_request(
                urls=self.request_endpoint,
                params=self.request_params,
            ).get(self.request_endpoint)
            or {}
        )

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

        result = (
            self.do_http_request(
                urls=self.request_endpoint,
                params=request_params,
            ).get(self.request_endpoint)
            or {}
        )

        return int(result.get("message", {}).get("total-results"))

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
                urls=self.request_endpoint,
                params=self.request_params,
            ).get(self.request_endpoint)

            if result is None:
                print("Found nothing.")
                return

            for item in result["message"]["items"]:
                yield item
            return

        else:
            request_params = dict(self.request_params)
            request_params["cursor"] = "*"
            request_params["rows"] = self.ROW_LIMIT
            while True:
                url = build_cr_endpoint(self.RESOURCE, self.context)
                result = self.do_http_request(
                    urls=url,
                    params=request_params,
                ).get(url)

                if result is None:
                    print("Found nothing.")
                    return

                if len(result["message"]["items"]) == 0:
                    print("Empty result.")
                    return
                else:
                    print(
                        f"Found {len(result['message']['items'])} items."
                    )
                for item in result["message"]["items"]:
                    yield item

                request_params["cursor"] = result["message"]["next-cursor"]

    def init_params(self) -> set[str]:
        """Get list of parameters for initialization."""

        return set(("email", "request_params", "context"))

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


class Crossref(Endpoint):
    """Wrap around the Crossref API."""

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

    def get_dois(
        self, dois: list[str]
    ) -> tuple[DataFrame, list[str]]:
        """
        Get all articles from a list of DOIs as dataframe.
        """

        # Get all articles from a list of DOIs
        urls = build_cr_endpoint(CrossrefResource.WORKS, endpoint=dois)
        raise NotImplementedError("This method is not implemented yet.")

    def get_refs(
        self, df: DataFrame, concurrent_lim: int = 50
    ) -> tuple[DataFrame, list[str]]:
        """
        Get all references from articles in the DataFrame.
        """

        # Get all references from articles in the DataFrame
        raise NotImplementedError("This method is not implemented yet.")
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

    # TODO: change it for using pydantic
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
            raise ValueError(
                f"Invalid filter {filter}. Valid filters: {self.VALIDATORS.keys()}"
            )
        try:
            return getattr(self, self.VALIDATORS[filter])(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for filter {filter}: {exc.args[0]}"
            ) from exc

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
    def is_integer(value: str | int) -> int:
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

        msg = "Directory specified as {} but must be one of: {}".format(
            str(value), ", ".join(expected)
        )
        raise ValueError(
            msg,
        )
