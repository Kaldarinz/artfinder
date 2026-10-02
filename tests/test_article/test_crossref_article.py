"""
Tests for turning a Crossref work record into an article DataFrame row.
"""

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from artfinder.article import ArticleCollection, CrossrefArticle, load_csv


def _record(**fields: Any) -> dict[str, Any]:
    """
    Build a minimal Crossref work record.

    Parameters
    ----------
    **fields : Any
        Crossref keys to add or override.

    Returns
    -------
    dict[str, Any]
        Work record as the Crossref API returns it.
    """

    record: dict[str, Any] = {
        "DOI": "10.1000/example",
        "title": ["An example"],
        "container-title": ["Example Journal"],
        "is-referenced-by-count": 0,
        "author": [{"given": "Jane", "family": "Doe", "sequence": "first"}],
    }
    record.update(copy.deepcopy(fields))
    return record


def _row(record: dict[str, Any]) -> pd.Series:
    """
    Convert one Crossref record to its DataFrame row.

    Parameters
    ----------
    record : dict[str, Any]
        Crossref work record.

    Returns
    -------
    pd.Series
        The article's row.
    """

    return CrossrefArticle(copy.deepcopy(record)).to_df().iloc[0]


def _csv_row(record: dict[str, Any], tmp_path: Path) -> pd.Series:
    """
    Convert one Crossref record to its row after a trip through a CSV file.

    Parameters
    ----------
    record : dict[str, Any]
        Crossref work record.
    tmp_path : Path
        Directory to write the CSV file to.

    Returns
    -------
    pd.Series
        The article's row as `load_csv` reads it back.
    """

    path = tmp_path / "articles.csv"
    ArticleCollection([copy.deepcopy(record)]).to_df().to_csv(path, index=False)
    return load_csv(str(path)).iloc[0]


class TestNoneInsideValues:
    """Text containing "none" must come back exactly as Crossref gave it."""

    RECORD = _record(
        author=[
            {
                "given": "Antonio",
                "family": "Iannone",
                "sequence": "first",
                "affiliation": [{"name": "Laboratory of nonequilibrium plasma"}],
            }
        ],
        subject=["Nonequilibrium Physics"],
    )

    def test_surname_is_kept(self) -> None:
        """A surname containing "none" is not rewritten to "None"."""
        assert _row(self.RECORD)["authors"][0]["lastname"] == "Iannone"

    def test_affiliation_is_kept(self) -> None:
        """An affiliation containing "none" is not rewritten to "None"."""
        assert _row(self.RECORD)["authors"][0]["affiliation"] == [
            "Laboratory of nonequilibrium plasma"
        ]

    def test_keyword_is_kept(self) -> None:
        """A keyword containing "none" stays lowercase."""
        assert _row(self.RECORD)["keywords"] == ["nonequilibrium physics"]

    def test_csv_round_trip_keeps_them(self, tmp_path: Path) -> None:
        """Parsing the columns back from a CSV file leaves the text alone."""
        row = _csv_row(self.RECORD, tmp_path)
        assert row["authors"][0]["lastname"] == "Iannone"
        assert row["authors"][0]["affiliation"] == [
            "Laboratory of nonequilibrium plasma"
        ]
        assert row["keywords"] == ["nonequilibrium physics"]


class TestListColumnsKeepCase:
    """List-valued columns keep Crossref's case; only reference DOIs are folded."""

    RECORD = _record(
        link=[
            {
                "URL": "https://api.elsevier.com/content/article/PII:S0169?httpAccept=text/xml",
                "content-type": "text/xml",
            }
        ],
        **{"issn-type": [{"value": "1474-175X", "type": "print"}]},
        funder=[{"name": "European Research Council", "award": ["ERC-2019-AdG"]}],
        reference=[{"DOI": "10.1021/ACS.JPCC.1"}],
    )

    def test_link_url_is_kept(self) -> None:
        """A URL's path and query are case-sensitive and stay as given."""
        assert (
            _row(self.RECORD)["links"][0]["url"]
            == "https://api.elsevier.com/content/article/PII:S0169?httpAccept=text/xml"
        )

    def test_issn_check_digit_is_kept(self) -> None:
        """An ISSN keeps its uppercase X check digit, as SciMagoJR lists it."""
        assert _row(self.RECORD)["issn"] == ["1474175X"]

    def test_funder_is_kept(self) -> None:
        """A funder's name and award number keep their case."""
        assert _row(self.RECORD)["funders"] == [
            {"name": "European Research Council", "number": "ERC-2019-AdG"}
        ]

    def test_reference_dois_are_lowercased(self) -> None:
        """DOIs are case-insensitive, and references are listed in lowercase."""
        assert _row(self.RECORD)["references"] == ["10.1021/acs.jpcc.1"]

    def test_csv_round_trip_keeps_case(self, tmp_path: Path) -> None:
        """Parsing the columns back from a CSV file keeps their case."""
        row = _csv_row(self.RECORD, tmp_path)
        assert row["issn"] == ["1474175X"]
        assert row["funders"][0]["name"] == "European Research Council"


def _heading(name: str) -> dict[str, Any]:
    """
    Build a Crossref author entry for a section heading or organisation.

    Parameters
    ----------
    name : str
        The heading.

    Returns
    -------
    dict[str, Any]
        Author entry carrying a bare name.
    """

    return {"name": name, "sequence": "additional", "affiliation": []}


def _person(family: str, given: str, *affiliations: str) -> dict[str, Any]:
    """
    Build a Crossref author entry for a person.

    Parameters
    ----------
    family : str
        Family name.
    given : str
        Given name.
    *affiliations : str
        Affiliation names.

    Returns
    -------
    dict[str, Any]
        Author entry.
    """

    return {
        "given": given,
        "family": family,
        "sequence": "additional",
        "affiliation": [{"name": name} for name in affiliations],
    }


# Entries 0-7, 31-33, 160 and 173-175 of the `author` array of 10.1038/nature15393:
# section headings, and people listed under each section they belong to.
SECTIONED_AUTHORS: list[dict[str, Any]] = [
    {**_heading("The 1000 Genomes Project Consortium"), "sequence": "first"},
    _heading("Corresponding authors"),
    _person("Auton", "Adam"),
    _person("Abecasis", "Gonçalo R."),
    _heading("Steering committee"),
    _person("Altshuler", "David M."),
    _person("Durbin", "Richard M."),
    _person("Abecasis", "Gonçalo R."),
    _heading("Production group"),
    _heading("Baylor College of Medicine"),
    _person("Gibbs", "Richard A."),
    _heading("Analysis group"),
    _heading("Baylor College of Medicine"),
    _person("Gibbs", "Richard A."),
    _person("Yu", "Fuli"),
]


class TestOrganisationSections:
    """An organisation and the members listed after it are not authors."""

    def test_byline_opened_by_a_consortium_has_no_authors(self) -> None:
        """Everyone after the consortium heading is one of its members."""
        assert _row(_record(author=SECTIONED_AUTHORS))["authors"] == []

    def test_people_before_the_organisation_are_kept(self) -> None:
        """The organisation, and every entry after it, is dropped."""
        record = _record(
            author=[
                {**_person("Doe", "Jane", "University A"), "sequence": "first"},
                _person("Wang", "Wei"),
                _heading("Example Consortium"),
                _person("Roe", "Richard"),
                _heading("Analysis group"),
                _person("Doe", "Jane"),
            ]
        )
        authors = _row(record)["authors"]
        assert [
            (author["lastname"], author["firstname"], author["affiliation"])
            for author in authors
        ] == [("Doe", "Jane", ["University A"]), ("Wang", "Wei", [])]

    def test_positions_follow_the_authors_kept(self) -> None:
        """Positions are derived as before, over the authors as returned."""
        record = _record(
            author=[
                {**_person("Doe", "Jane"), "sequence": "first"},
                _person("Wang", "Wei"),
                _person("Roe", "Richard"),
                _heading("Example Consortium"),
                _person("Poe", "Edgar"),
            ]
        )
        positions = [author["position"] for author in _row(record)["authors"]]
        assert positions == ["first", "additional", "last"]

    def test_homonyms_are_kept(self) -> None:
        """Two authors sharing a name are two authors."""
        record = _record(
            author=[
                _person("Wang", "Wei", "Tsinghua University"),
                _person("Doe", "Jane"),
                _person("Wang", "Wei", "Chinese Academy of Sciences"),
            ]
        )
        authors = _row(record)["authors"]
        assert [
            (author["lastname"], author["affiliation"]) for author in authors
        ] == [
            ("Wang", ["Tsinghua University"]),
            ("Doe", []),
            ("Wang", ["Chinese Academy of Sciences"]),
        ]

    def test_csv_round_trip_keeps_missing_firstname(self, tmp_path: Path) -> None:
        """A person's missing firstname reads back as None."""
        record = _record(author=[{"family": "Doe", "sequence": "first"}])
        authors = _csv_row(record, tmp_path)["authors"]
        assert authors[0]["lastname"] == "Doe"
        assert authors[0]["firstname"] is None

    def test_record_is_not_modified(self) -> None:
        """Extracting the authors leaves the Crossref record as it was."""
        record = _record(author=SECTIONED_AUTHORS)
        CrossrefArticle(record)
        assert record["author"] == SECTIONED_AUTHORS


class TestFunderAwards:
    """Each award of a funder becomes a funder record of its own."""

    def test_two_awards_give_two_records(self) -> None:
        """Both awards are kept, in Crossref's order, as deposited."""
        record = _record(
            funder=[
                {
                    "name": "Russian Science Foundation",
                    "DOI": "10.13039/501100006769",
                    "award": ["20-72-00081", "no. FSWU-2020-0035"],
                }
            ]
        )
        assert _row(record)["funders"] == [
            {
                "name": "Russian Science Foundation",
                "doi": "10.13039/501100006769",
                "number": "20-72-00081",
            },
            {
                "name": "Russian Science Foundation",
                "doi": "10.13039/501100006769",
                "number": "no. FSWU-2020-0035",
            },
        ]

    def test_one_award_gives_one_record(self) -> None:
        """A single award gives a single record, its backslashes kept."""
        record = _record(
            funder=[{"name": "Royal Society", "award": ["rsrp\\r\\190000"]}]
        )
        assert _row(record)["funders"] == [
            {"name": "Royal Society", "number": "rsrp\\r\\190000"}
        ]

    def test_no_awards_give_one_record_without_number(self) -> None:
        """A funder with a missing or empty award list is listed once."""
        record = _record(
            funder=[
                {"name": "National Science Foundation"},
                {"name": "European Research Council", "award": []},
            ]
        )
        assert _row(record)["funders"] == [
            {"name": "National Science Foundation"},
            {"name": "European Research Council"},
        ]


class TestAbstract:
    def test_every_section_body_is_kept(self) -> None:
        """Removing section titles leaves the text between them."""
        record = _record(
            abstract=(
                "<jats:title>Background</jats:title><jats:p>Lasers ablate.</jats:p>"
                "<jats:title>Results</jats:title><jats:p>Particles form.</jats:p>"
            )
        )
        assert _row(record)["abstract"] == "lasers ablate.particles form."

    def test_multiline_title_is_removed(self) -> None:
        """A title broken over lines is removed with its text."""
        record = _record(
            abstract="<jats:title>\nAbstract\n</jats:title>\n<jats:p>Lasers ablate.</jats:p>"
        )
        assert _row(record)["abstract"] == "lasers ablate."
