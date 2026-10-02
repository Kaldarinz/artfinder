"""
Tests for carrying settings through `Crossref`'s chained query methods.
"""

from artfinder.crossref import Crossref


def test_chaining_keeps_print_status() -> None:
    """A chained query keeps the `print_status` it was created with."""

    crossref = (
        Crossref(print_status=False)
        .search("x")
        .author("y")
        .filter(from_pub_date="2020")
        .article()
    )

    assert crossref.print_status is False
