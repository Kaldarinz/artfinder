"""
Tests for normalizing a DOI as a person or a publisher writes it.
"""

import pytest

from artfinder import normalize_doi

DOI = "10.1016/j.apsusc.2019.144012"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (DOI, DOI),
        ("10.1016/J.APSUSC.2019.144012", DOI),
        (f"doi:{DOI}", DOI),
        (f"DOI: {DOI}", DOI),
        (f"https://doi.org/{DOI}", DOI),
        (f"http://dx.doi.org/{DOI}", DOI),
        (f"doi.org/{DOI}", DOI),
        (f"  {DOI}\n", DOI),
        (f"{DOI}.", DOI),
        (f"https://doi.org/{DOI});", DOI),
        ("https://doi.org/10.1002%2Fadom.201800164", "10.1002/adom.201800164"),
        ("10.1073/pnas.2208830119/-/DCSupplemental", "10.1073/pnas.2208830119"),
        ("10.1016/0021-9797(80)90001-7", "10.1016/0021-9797(80)90001-7"),
        ("", None),
        ("   ", None),
        ("not a doi", None),
        ("1000/abc.123", None),
        ("10.1016", None),
        ("10.12/abc", None),
        (f"see {DOI}", None),
        ("10.1016/j.apsusc 2019", None),
        (f"https://example.org/{DOI}", None),
    ],
)
def test_normalize_doi(text: str, expected: str | None) -> None:
    """Prefixes, case, whitespace and trailing punctuation go; a non-DOI is None."""
    assert normalize_doi(text) == expected
