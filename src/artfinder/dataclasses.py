# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT

from typing import NamedTuple
from enum import StrEnum



class CrossrefRateLimit(NamedTuple):
    """
    A class to represent the Crossref rate limit.
    """

    limit: int
    interval: int

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

class CrossrefSelectField(StrEnum):
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

class CrossrefFilterField(StrEnum):
    """
    Enum for Crossref filter fields.
    """
    # Members generated from VALIDATORS keys
    ALTERNATIVE_ID = "alternative-id"
    ARCHIVE = "archive"
    ARTICLE_NUMBER = "article-number"
    ASSERTION = "assertion"
    ASSERTION_GROUP = "assertion-group"
    AWARD_FUNDER = "award.funder"
    AWARD_NUMBER = "award.number"
    CATEGORY_NAME = "category-name"
    CLINICAL_TRIAL_NUMBER = "clinical-trial-number"
    CONTAINER_TITLE = "container-title"
    CONTENT_DOMAIN = "content-domain"
    DIRECTORY = "directory"
    DOI = "doi"
    FROM_ACCEPTED_DATE = "from-accepted-date"
    FROM_CREATED_DATE = "from-created-date"
    FROM_DEPOSIT_DATE = "from-deposit-date"
    FROM_EVENT_END_DATE = "from-event-end-date"
    FROM_EVENT_START_DATE = "from-event-start-date"
    FROM_INDEX_DATE = "from-index-date"
    FROM_ISSUED_DATE = "from-issued-date"
    FROM_ONLINE_PUB_DATE = "from-online-pub-date"
    FROM_POSTED_DATE = "from-posted-date"
    FROM_PRINT_PUB_DATE = "from-print-pub-date"
    FROM_PUB_DATE = "from-pub-date"
    FROM_UPDATE_DATE = "from-update-date"
    FULL_TEXT_APPLICATION = "full-text.application"
    FULL_TEXT_TYPE = "full-text.type"
    FULL_TEXT_VERSION = "full-text.version"
    FUNDER = "funder"
    FUNDER_DOI_ASSERTED_BY = "funder-doi-asserted-by"
    GROUP_TITLE = "group-title"
    HAS_ABSTRACT = "has-abstract"
    HAS_AFFILIATION = "has-affiliation"
    HAS_ARCHIVE = "has-archive"
    HAS_ASSERTION = "has-assertion"
    HAS_AUTHENTICATED_ORCID = "has-authenticated-orcid"
    HAS_AWARD = "has-award"
    HAS_CLINICAL_TRIAL_NUMBER = "has-clinical-trial-number"
    HAS_CONTENT_DOMAIN = "has-content-domain"
    HAS_DOMAIN_RESTRICTION = "has-domain-restriction"
    HAS_EVENT = "has-event"
    HAS_FULL_TEXT = "has-full-text"
    HAS_FUNDER = "has-funder"
    HAS_FUNDER_DOI = "has-funder-doi"
    HAS_LICENSE = "has-license"
    HAS_ORCID = "has-orcid"
    HAS_REFERENCES = "has-references"
    HAS_RELATION = "has-relation"
    HAS_UPDATE = "has-update"
    HAS_UPDATE_POLICY = "has-update-policy"
    IS_UPDATE = "is-update"
    ISBN = "isbn"
    ISSN = "issn"
    LICENSE_DELAY = "license.delay"
    LICENSE_URL = "license.url"
    LICENSE_VERSION = "license.version"
    LOCATION = "location"
    MEMBER = "member"
    ORCID = "orcid"
    PREFIX = "prefix"
    RELATION_OBJECT = "relation.object"
    RELATION_OBJECT_TYPE = "relation.object-type"
    RELATION_TYPE = "relation.type"
    TYPE = "type"
    TYPE_NAME = "type-name"
    UNTIL_ACCEPTED_DATE = "until-accepted-date"
    UNTIL_CREATED_DATE = "until-created-date"
    UNTIL_DEPOSIT_DATE = "until-deposit-date"
    UNTIL_EVENT_END_DATE = "until-event-end-date"
    UNTIL_EVENT_START_DATE = "until-event-start-date"
    UNTIL_INDEX_DATE = "until-index-date"
    UNTIL_ISSUED_DATE = "until-issued-date"
    UNTIL_ONLINE_PUB_DATE = "until-online-pub-date"
    UNTIL_POSTED_DATE = "until-posted-date"
    UNTIL_PRINT_PUB_DATE = "until-print-pub-date"
    UNTIL_PUB_DATE = "until-pub-date"
    UNTIL_UPDATE_DATE = "until-update-date"
    UPDATE_TYPE = "update-type"
    UPDATES = "updates"


class DocumentType(StrEnum):
    """Valid document types."""

    BOOK = 'book'
    BOOK_CHAPTER = 'book-chapter'
    BOOK_SET = 'book-set'
    BOOK_SERIES = 'book-series'
    BOOK_PART = 'book-part'
    BOOK_SECTION = 'book-section'
    BOOK_TRACK = 'book-track'
    REFERENCE_BOOK = 'reference-book'
    EDITED_BOOK = 'edited-book'
    MONOGRAPH = 'monograph'
    REPORT = 'report'
    PROCEEDINGS = 'proceedings'
    PROCEEDINGS_ARTICLE = 'proceedings-article'
    JOURNAL = 'journal'
    JOURNAL_ARTICLE = 'journal-article'
    JOURNAL_VOLUME = 'journal-volume'
    JOURNAL_ISSUE = 'journal-issue'
    OTHER = 'other'
    REFERENCE_ENTRY = 'reference-entry'
    COMPONENT = 'component'
    REPORT_SERIES = 'report-series'
    STANDARD = 'standard'
    STANDARD_SERIES = 'standard-series'
    POSTER_CONTENT = 'poster-content'
    DISSERTATION = 'dissertation'
    DATASET = 'dataset'