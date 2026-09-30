# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
Helper functions and constants for interacting with the Crossref API.

This module is a part of the Artfinder package.
"""

import logging
import re

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

FUNDER_REGISTRY_ID_PATTERN = re.compile(r"10\.13039/([^/\s]+)$", re.IGNORECASE)
"Funder Registry DOI, bare or as a URL; the group is the funder id."


def funder_registry_id(doi: str | None) -> str | None:
    """
    Get the Funder Registry id from a funder DOI.

    Parameters
    ----------
    doi : str | None
        Funder DOI as deposited, e.g. `10.13039/501100001823` or
        `https://doi.org/10.13039/501100001823`.

    Returns
    -------
    str | None
        The id, `501100001823`, or None when the DOI is not a Funder Registry one.
    """

    if not doi:
        return None
    match = FUNDER_REGISTRY_ID_PATTERN.search(doi.strip())
    return match.group(1) if match else None
