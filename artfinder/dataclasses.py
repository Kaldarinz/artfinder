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
    Crossref query fields.
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

class CrossrefFilterField(StrEnum):
    """
    Valid Crossref filter fields.
    """

    DOI = 'DOI'
    ISBN = 'ISBN'
    ISSN = 'ISSN'
    URL = 'URL'
    ABSTRACT = 'abstract'
    ACCEPTED = 'accepted'
    ALTERNATIVE_ID = 'alternative-id'
    APPROVED = 'approved'
    ARCHIVE = 'archive'
    ARTICLE_NUMBER = 'article-number'
    ASSERTION = 'assertion'
    AUTHOR = 'author'
    CHAIR = 'chair'
    CLINICAL_TRIAL_NUMBER = 'clinical-trial-number'
    CONTAINER_TITLE = 'container-title'
    CONTENT_CREATED = 'content-created'
    CONTENT_DOMAIN = 'content-domain'
    CREATED = 'created'
    DEGREE = 'degree'
    DEPOSITED = 'deposited'
    EDITOR = 'editor'
    EVENT = 'event'
    FUNDER = 'funder'
    GROUP_TITLE = 'group-title'
    INDEXED = 'indexed'
    IS_REFERENCED_BY_COUNT = 'is-referenced-by-count'
    ISSN_TYPE = 'issn-type'
    ISSUE = 'issue'
    ISSUED = 'issued'
    LICENSE = 'license'
    LINK = 'link'
    MEMBER = 'member'
    ORIGINAL_TITLE = 'original-title'
    PAGE = 'page'
    POSTED = 'posted'
    PREFIX = 'prefix'
    PUBLISHED = 'published'
    PUBLISHED_ONLINE = 'published-online'
    PUBLISHED_PRINT = 'published-print'
    PUBLISHER = 'publisher'
    PUBLISHER_LOCATION = 'publisher-location'
    REFERENCE = 'reference'
    REFERENCES_COUNT = 'references-count'
    RELATION = 'relation'
    SCORE = 'score'
    SHORT_CONTAINER_TITLE = 'short-container-title'
    SHORT_TITLE = 'short-title'
    STANDARDS_BODY = 'standards-body'
    SUBJECT = 'subject'
    SUBTITLE = 'subtitle'
    TITLE = 'title'
    TYPE = 'type'
    TRANSLATOR = 'translator'
    UPDATE_POLICY = 'update-policy'
    UPDATE_TO = 'update-to'
    UPDATED_BY = 'updated-by'
    VOLUME = 'volume'
