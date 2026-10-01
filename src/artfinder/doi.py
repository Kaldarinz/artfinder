# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT

"""DOI parsing shared by the PDF reader and the callers that validate user input."""

import re
from urllib.parse import unquote

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
"Pattern of a DOI in text. The prefix dot is literal: `1000/x` is not a DOI."
DOI_PREFIX_PATTERN = re.compile(
    r"^(?:doi:\s*|(?:https?://)?(?:dx\.)?doi\.org/)", re.IGNORECASE
)
"Pattern of what a DOI is written after: `doi:` or a doi.org resolver URL."


def strip_doi_decorations(candidate: str) -> str:
    """
    Strip the decorations publishers append to a DOI in running text.

    Parameters
    ----------
    candidate : str
        Raw DOI match.

    Returns
    -------
    str
        Lowercased DOI without a supplemental-material suffix or trailing
        sentence punctuation.
    """

    # ".../-/DCSupplemental" and friends address the supplement, not the article.
    return re.split(r"/-/", candidate.lower())[0].rstrip(".,;:)")


def normalize_doi(doi: str) -> str | None:
    """
    Reduce a DOI as a person or a publisher writes it to the bare DOI.

    Accepts a bare DOI, one written after `doi:`, or a `doi.org` / `dx.doi.org`
    resolver URL, with surrounding whitespace and trailing punctuation. The
    result is in the form `ArticlePDF.doi` returns.

    Parameters
    ----------
    doi : str
        DOI to normalize, e.g. `https://doi.org/10.1016/j.apsusc.2019.144012`.

    Returns
    -------
    str | None
        Lowercased bare DOI, e.g. `10.1016/j.apsusc.2019.144012`, or None when
        the text is not a DOI.
    """

    text = doi.strip()
    prefix = DOI_PREFIX_PATTERN.match(text)
    if prefix is not None:
        text = text[prefix.end() :]
        # A resolver URL may carry the DOI percent-encoded.
        if "doi.org" in prefix.group().lower():
            text = unquote(text)
    text = strip_doi_decorations(text)
    return text if DOI_PATTERN.fullmatch(text) else None
