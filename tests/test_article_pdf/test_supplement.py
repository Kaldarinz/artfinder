"""
Tests for telling a supplementary-material PDF from an article's main text.
"""

import re
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from artfinder.article_pdf import ArticlePDF

TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"

SI_NAME = re.compile(r"(?<!_with)_si(?:_watermarked)?\.pdf$")
"File name of a supplement in the corpus; `_with_si` is a main text bound with its SI."
UNDETECTED = {
    # Opens directly on "Table S1. Humoral factors …", with no heading at all.
    "silicon-gold nanoparticles affect wharton's jelly phenotype and secretome"
    " during tri-lineage differentiation_si.pdf",
}
"Supplements neither clause catches, documented as such."

ASSOCIATED_CONTENT_SI = (
    "laser fragmentation of colloidal gold nanoparticles with high-intensity"
    " nanosecond pulses_si.pdf"
)
SPRINGER_FOOTNOTE = (
    "nanocomposites composed of p3ht_pcbm and nanoparticles synthesized by laser"
    " ablation of a bulk pbs target in liquid.pdf"
)
WITH_SI = (
    "whole-cell patch-clamp measurements of spermatozoa reveal an"
    " alkaline-activated ca channel_with_si.pdf"
)
S_PAGINATED = [
    "bi@sio2 core@shell composites formation based on laser synthesized bi"
    " nanoparticles.pdf",
    "control of dimensional and optical properties of laser-synthesized tin"
    " nanoparticles for biomedical applications.pdf",
    "study of efficiency of formation of core–satellite gold- and iron-based"
    " nanostructures by laser ablation synthesis.pdf",
    "the effect of pulse duration on properties of boron nanoparticles produced by"
    " laser fragmentation of micropowders in liquids.pdf",
    "x-ray contrast properties of bismuth-based nanoformulations.pdf",
]
ACS_MAIN_TEXT = "tunable nanostructuring for van der waals materials.pdf"
RSC_MAIN_TEXT = (
    "tunable optical properties of transition metal dichalcogenide nanoparticles"
    " synthesized by femtosecond laser ablation and fragmentation.pdf"
)

BODY_LINE = "Laser ablation of a target in water gives colloidal nanoparticles."
RSC_STAMP = "Electronic Supplementary Material (ESI) for Journal of Materials Chemistry C"


def _is_supplement(file_name: str) -> bool:
    """
    Read `is_supplement` of a corpus PDF.

    Parameters
    ----------
    file_name : str
        File name under `article_pdfs/`.

    Returns
    -------
    bool
        The PDF's `is_supplement`.
    """

    with ArticlePDF(TEST_PDFS_DIR / file_name) as pdf:
        return pdf.is_supplement


def _page_of_lines(lines: list[tuple[float, str, float]]) -> bytes:
    """
    Build a one-page PDF of Helvetica text lines.

    Parameters
    ----------
    lines : list[tuple[float, str, float]]
        Baseline position from the top of the page, text and font size of each line.

    Returns
    -------
    bytes
        The PDF.
    """

    doc = pymupdf.open()
    # PyMuPDF's `Page` is invisible to mypy (docs/test_setup.md).
    page: Any = doc.new_page()
    for y, text, size in lines:
        page.insert_text((72, y), text, fontsize=size, fontname="helv")
    return doc.tobytes()


def _body(count: int, start: float = 72) -> list[tuple[float, str, float]]:
    """
    Lines of 9 pt body text, one every 14 pt.

    Parameters
    ----------
    count : int
        Number of lines.
    start : float, optional
        Baseline of the first line.

    Returns
    -------
    list[tuple[float, str, float]]
        Lines as `_page_of_lines` takes them.
    """

    return [(start + 14 * i, BODY_LINE, 9) for i in range(count)]


@pytest.mark.parametrize(
    "line",
    [
        # Nature-style openings, the full word and the abbreviations.
        "Supplementary Figure 1 | SEM images",
        "Supplementary Fig. 1 | SEM images",
        "Supplementary Figs. 1–3",
        "Supplementary Fig 1",
        "Supplementary Figures",
        "Supplementary Table S1",
        "Supplementary Tables",
        "Supplementary Note 1",
        "Supplementary Methods",
        "Supplementary Appendix",
        "Supplementary Appendices",
        "Online Resource 1",
        "Associated content",
        "ASSOCIATED CONTENT",
        # What SUPPLEMENT_HEADING_PATTERN already covers.
        RSC_STAMP,
        "Supporting Information",
        "Supporting   Information",
        "Supplementary Materials: Laser-Ablative Synthesis of",
        "Supplemental Data",
    ],
)
def test_detection_pattern_matches(line: str) -> None:
    """A line opening a supplement matches from its start."""
    assert ArticlePDF.SUPPLEMENT_DETECTION_PATTERN.match(line)


@pytest.mark.parametrize(
    "line",
    [
        # Real lines of corpus main texts.
        "Supporting localized light-induced plasmon oscil-",
        "Capable of supporting collective oscillations of free electrons",
        # Not at the start.
        "see Supplementary Figure 1",
        # A word must follow "Supplementary", and end where it should.
        "Supplementary",
        "Supplementary Figx",
        "Supplementary Tablet",
        "Table S1. Humoral factors measured in MSCs supernatants",
    ],
)
def test_detection_pattern_rejects(line: str) -> None:
    """Text that merely mentions supplementary material does not match."""
    assert ArticlePDF.SUPPLEMENT_DETECTION_PATTERN.match(line) is None


def test_heading_pattern_is_not_widened() -> None:
    """The pattern titles are stripped with still leaves "Supplementary Figure" alone."""
    assert ArticlePDF.SUPPLEMENT_HEADING_PATTERN.match("Supplementary Figure 1 | x") is None


@pytest.mark.parametrize(
    "file_name", sorted(path.name for path in TEST_PDFS_DIR.glob("*.pdf"))
)
def test_corpus(file_name: str) -> None:
    """Every supplement in the corpus but the documented ones is detected, no main text is."""
    expected = bool(SI_NAME.search(file_name)) and file_name not in UNDETECTED
    assert _is_supplement(file_name) is expected


def test_associated_content_heading() -> None:
    """ACS "Associated content" at 16 pt over a 12 pt body is a heading; the SI carries
    its parent's DOI."""
    with ArticlePDF(TEST_PDFS_DIR / ASSOCIATED_CONTENT_SI) as pdf:
        assert pdf._has_supplement_heading()
        assert pdf.doi == "10.1021/acs.jpcc.8b04374"


def test_springer_footnote() -> None:
    """Springer's 8.5 pt "Electronic supplementary material" footnote, two thirds down
    page 1, matches the pattern but is no heading."""
    assert _is_supplement(SPRINGER_FOOTNOTE) is False


def test_main_text_bound_with_si() -> None:
    """A main text with its SI bound behind it is a main text."""
    assert _is_supplement(WITH_SI) is False


@pytest.mark.parametrize("file_name", S_PAGINATED)
def test_s_paginated_main_texts(file_name: str) -> None:
    """Pages numbered S594 and the like, in a journal supplement issue, mean nothing."""
    assert _is_supplement(file_name) is False


def test_acs_link_line() -> None:
    """The 8 pt "Supporting Information" link line of an ACS main text is no heading."""
    assert _is_supplement(ACS_MAIN_TEXT) is False


def test_rsc_esi_footnote() -> None:
    """An RSC main text's "† Electronic supplementary information (ESI) available"
    footnote is no heading."""
    assert _is_supplement(RSC_MAIN_TEXT) is False


@pytest.mark.parametrize(("position", "expected"), [(0, True), (2, False)])
def test_opening_line(position: int, expected: bool) -> None:
    """An RSC ESI stamp at body size counts only as one of the first two lines."""
    lines = _body(20)
    lines.insert(position, (0.0, RSC_STAMP, 9))
    data = _page_of_lines([(72 + 14 * i, text, size) for i, (_, text, size) in enumerate(lines)])
    with ArticlePDF(data) as pdf:
        assert pdf._opens_with_supplement_line() is expected
        assert pdf._has_supplement_heading() is False
        assert pdf.is_supplement is expected


@pytest.mark.parametrize(("y", "expected"), [(120, True), (600, False)])
def test_heading(y: float, expected: bool) -> None:
    """A heading larger than body text counts only in the top half of the page."""
    data = _page_of_lines(_body(20) + [(y, "Supporting Information", 16)])
    with ArticlePDF(data) as pdf:
        assert pdf._opens_with_supplement_line() is False
        assert pdf._has_supplement_heading() is expected
        assert pdf.is_supplement is expected


@pytest.mark.parametrize("pages", [1, 3])
def test_scan(pages: int) -> None:
    """An image-only PDF with its DOI in the metadata reads without raising."""
    source = pymupdf.open()
    text_page: Any = source.new_page()
    text_page.insert_text(
        (72, 100), "Laser ablation synthesis of Au nanoparticles in acetone", fontsize=16
    )
    pixmap = text_page.get_pixmap(dpi=100)
    doc = pymupdf.open()
    for _ in range(pages):
        page: Any = doc.new_page()
        page.insert_image(page.rect, pixmap=pixmap)
    doc.set_metadata({"subject": "doi:10.5555/scan.2019.001"})
    with ArticlePDF(doc.tobytes()) as pdf:
        assert pdf.doi == "10.5555/scan.2019.001"
        assert pdf.title is None
        assert pdf.is_supplement is False
