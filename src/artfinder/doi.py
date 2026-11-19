# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT

"""DOI parsing shared by the PDF reader and the callers that validate user input."""

import re
from collections.abc import Iterator
from urllib.parse import unquote

_DOI_REGISTRANT = r"\d{4,9}/"
_DOI_SUFFIX = r"[-._;()/:A-Z0-9]+"
DOI_PATTERN = re.compile(rf"10\.{_DOI_REGISTRANT}{_DOI_SUFFIX}", re.IGNORECASE)
"Pattern of a DOI in text. The prefix dot is literal: `1000/x` is not a DOI."
DOI_PREFIX_PATTERN = re.compile(
    r"^(?:doi:\s*|(?:https?://)?(?:dx\.)?doi\.org/)", re.IGNORECASE
)
"Pattern of what a DOI is written after: `doi:` or a doi.org resolver URL."

_LINE_BREAK = r"[ \t]*\r?\n[ \t]*"
_TEXT_DOI_START = re.compile(
    rf"10\.(?:{_LINE_BREAK})?{_DOI_REGISTRANT}(?P<suffix>{_DOI_SUFFIX})?",
    re.IGNORECASE,
)
_TEXT_DOI_CONTINUATION = re.compile(
    rf"{_LINE_BREAK}({_DOI_SUFFIX})(?=,?[ \t]*\r?$)",
    re.IGNORECASE | re.MULTILINE,
)


def iter_dois_in_text(text: str) -> Iterator[str]:
    """
    Yield normalized DOI matches, recovering likely line wraps.

    A bare registrant (e.g. ``10.1234/``) needs a suffix. Once a suffix is
    present, extend it after ``.``, ``/`` or ``-`` only if the next line is a
    single DOI-shaped fragment containing a digit. A word on its own could be
    either a suffix fragment or a heading; keep the DOI already found rather
    than guessing. An isolated number can still look like a continuation.

    Parameters
    ----------
    text : str
        Text to search, preserving its line breaks.

    Returns
    -------
    Iterator[str]
        Normalized DOI matches in text order.
    """

    pos = 0
    while start := _TEXT_DOI_START.search(text, pos):
        pos = start.end()
        raw = "".join(start.group().split())
        has_suffix = start["suffix"] is not None
        while raw.endswith((".", "/", "-")) and (
            continuation := _TEXT_DOI_CONTINUATION.match(text, pos)
        ):
            fragment = continuation[1]
            if _TEXT_DOI_START.match(fragment):
                break  # The next line begins a new DOI.
            if has_suffix and not any(char.isdigit() for char in fragment):
                break
            raw += fragment
            pos = continuation.end()
            has_suffix = True
        if doi := normalize_doi(raw):
            yield doi


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
