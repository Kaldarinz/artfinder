"""
AsyncHTTP request class for making requests.

This module is part of the Artfinder package.
Author: Anton Popov
email: a.popov.fizteh@gmail.com
"""

import contextlib
import logging
import requests
from time import sleep


logger = logging.getLogger(__name__)

class HTTPRequest:

    def __init__(self, throttle=True):
        self.throttle = throttle
        self.rate_limits = {"x-rate-limit-limit": 50, "x-rate-limit-interval": 1}

    def _update_rate_limits(self, headers):

        with contextlib.suppress(ValueError):
            self.rate_limits["x-rate-limit-limit"] = int(headers.get("x-rate-limit-limit", 50))

        with contextlib.suppress(ValueError):
            interval_value = int(headers.get("x-rate-limit-interval", "1s")[:-1])

        interval_scope = headers.get("x-rate-limit-interval", "1s")[-1]

        if interval_scope == "m":
            interval_value = interval_value * 60

        if interval_scope == "h":
            interval_value = interval_value * 60 * 60

        self.rate_limits["x-rate-limit-interval"] = interval_value

    @property
    def throttling_time(self):
        return self.rate_limits["x-rate-limit-interval"] / self.rate_limits["x-rate-limit-limit"]

    def do_http_request(  # noqa: PLR0913
            self,
            method,
            endpoint,
            only_headers,
            data=None,
            files=None,
            timeout=100,
            custom_header=None,
    ):

        if only_headers is True:
            logger.info(f"HEAD request to {endpoint}")
            return requests.head(endpoint, timeout=2)

        action = requests.post if method == "post" else requests.get

        if method == "post":
            result = action(endpoint, data=data, files=files, timeout=timeout, headers=custom_header)
        else:
            result = action(endpoint, params=data, timeout=timeout, headers=custom_header)

        if self.throttle is True:
            self._update_rate_limits(result.headers)
            sleep(self.throttling_time)

        return result
