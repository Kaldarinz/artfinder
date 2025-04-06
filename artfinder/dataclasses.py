from typing import NamedTuple
from enum import StrEnum

CROSSREF_API_BASE = 'api.crossref.org'

class CrossrefRateLimit(NamedTuple):
    """
    A class to represent the Crossref rate limit.
    """

    limit: str
    interval: str

class CrossrefResource(StrEnum):
    """
    A class to represent the Crossref API endpoints.
    """

    WORKS = 'works'
    FUNDERS = 'funders'
    MEMBERS = 'members'
    PREFIXES = 'prefixes'
    TYPES = 'types'
    JOURNALS = 'journals'


class CrossrefQueryField(StrEnum):
    """
    A class to represent the Crossref API query fields.
    """

    AFFILIATION = 'affiliation'
    AUTHOR = 'author'
    BIBLIOGRAPHIC = 'bibliographic'
    CHAIR = 'chair'
    CONTAINER_TITLE = 'container-title'
    CONTRIBUTOR = 'contributor'
    EDITOR = 'editor'
    EVENT_ACRONYM = 'event-acronym'
    EVENT_LOCATION = 'event-location'
    EVENT_NAME = 'event-name'
    EVENT_SPONSOR = 'event-sponsor'
    EVENT_THEME = 'event-theme'
    FUNDER = 'funder-name'
    PUBLICHER = 'publisher-name'
    PUBLISHER_LOCATION = 'publisher-location'
    TRANSLATOR = 'translator'