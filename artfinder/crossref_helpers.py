"""
Helper functions and constants for interacting with the Crossref API.

This module is a part of the Artfinder package.
Author: Anton Popov
email: a.popov.fizteh@gmail.com
"""

import logging

from artfinder.dataclasses import CrossrefResource

logger = logging.getLogger(__name__)

CROSSREF_API_BASE = 'api.crossref.org'

def build_cr_endpoint(
    resource: CrossrefResource,
    endpoint: str | list[str] | None = None,
    context: str | list[str] | None = None,
) -> str:
    """
    Build the Crossref API endpoint URL.

    Parameters
    ----------
    resource : CrossrefResource
        The base resource for the Crossref API (e.g., "works", "funders").
    endpoint : str | list[str] | None, optional
        The specific endpoint(s) to access. Can be a single string or a list of strings.
    context : str | list[str] | None, optional
        Additional path segments to provide context for the endpoint.

    Returns
    -------
        The complete URL for the Crossref API endpoint.

    Examples
    --------
    >>> build_cr_endpoint("works", "10.1000/xyz123")
    'https://api.crossref.org/works/10.1000/xyz123'

    >>> build_cr_endpoint("funders", ["10.13039", "501100000735"])
    'https://api.crossref.org/funders/10.13039/501100000735'

    >>> build_cr_endpoint("works", context=["journals"])
    'https://api.crossref.org/journals/works'
    """

    # Ensure endpoint and context are lists
    if endpoint and not isinstance(endpoint, list):
        endpoint = [endpoint]
    if context and not isinstance(context, list):
        context = [context]

    # Construct the endpoint path
    parts = []
    if context:
        parts.extend(context)
    parts.append(resource)
    if endpoint:
        parts.extend(endpoint)

    endpoint_path = "/".join(parts)
    complete_url = f"https://{CROSSREF_API_BASE}/{endpoint_path}"

    return complete_url