"""
Tests for writing a DOI into a PDF's metadata.
"""

import shutil
from pathlib import Path

import pytest

from artfinder.article_pdf import ArticlePDF

TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"

DOI = "10.1016/j.apsusc.2019.144012"

# An XMP packet without a DOI.
NO_DOI_PDF = (
    TEST_PDFS_DIR
    / "bismuth nanoparticles increase effectiveness of proton therapy of ehrlich"
    " carcinoma.pdf"
)
# No XMP packet at all.
NO_XMP_PDF = (
    TEST_PDFS_DIR
    / "bare laser-synthesized si nanoparticles as functional elements for chitosan"
    " nanofiber-based tissue engineering platforms.pdf"
)
# The DOI in the subject, and in the XMP packet as prism:doi, dc:identifier,
# prism:url and the description.
ELSEVIER_PDF = (
    TEST_PDFS_DIR
    / "localized infrared radiation-induced hyperthermia sensitized by laser-ablated"
    " silicon nanoparticles for phototherapy applications.pdf"
)
ELSEVIER_DOI = "10.1016/j.apsusc.2020.145661"


def _copy(src: Path, tmp_path: Path) -> Path:
    """
    Copy a test PDF into the test's temporary directory.

    Parameters
    ----------
    src : Path
        PDF to copy.
    tmp_path : Path
        Pytest's temporary directory.

    Returns
    -------
    Path
        The copy.
    """

    dst = tmp_path / "article.pdf"
    shutil.copy(src, dst)
    return dst


def _page_contents(pdf: ArticlePDF) -> list[bytes]:
    """
    Read the content stream of every page.

    Parameters
    ----------
    pdf : ArticlePDF
        Open PDF.

    Returns
    -------
    list[bytes]
        Content of each page, in page order.
    """

    pages = range(pdf.file.page_count)
    return [pdf._page(page_no).read_contents() for page_no in pages]


def test_round_trip_to_dst(tmp_path: Path) -> None:
    """The DOI is read back from the saved copy; the source is left as it was."""
    src = _copy(NO_DOI_PDF, tmp_path)
    original = src.read_bytes()
    dst = tmp_path / "with_doi.pdf"
    with ArticlePDF(src) as pdf:
        contents = _page_contents(pdf)
        pdf.set_doi(f"https://doi.org/{DOI}", dst)
        assert pdf.doi == DOI

    assert src.read_bytes() == original
    with ArticlePDF(dst) as pdf:
        assert pdf.doi == DOI
        assert pdf.file.metadata["subject"] == f"doi:{DOI}"
        assert pdf._extract_doi_from_xmp() == DOI
        assert _page_contents(pdf) == contents


def test_existing_xmp_fields_survive(tmp_path: Path) -> None:
    """The DOI is merged into the packet, not written over it."""
    dst = tmp_path / "with_doi.pdf"
    with ArticlePDF(NO_DOI_PDF) as pdf:
        before = pdf._xmp_packet()
        pdf.set_doi(DOI, dst)

    with ArticlePDF(dst) as pdf:
        after = pdf._xmp_packet()
    for field in [
        "<xmp:CreatorTool>Adobe InDesign 16.0 (Windows)</xmp:CreatorTool>",
        "<xmpMM:DocumentID>uuid:29231ddd-07de-4570-b5f9-3085a61ef73c</xmpMM:"
        "DocumentID>",
        "<stEvt:parameters>converted to PDF/A-2b</stEvt:parameters>",
    ]:
        assert field in before
        assert field in after
    assert f"<prism:doi>{DOI}</prism:doi>" in after
    assert f"<dc:identifier>doi:{DOI}</dc:identifier>" in after


def test_pdf_without_xmp_gets_a_packet(tmp_path: Path) -> None:
    """A PDF with no XMP packet gets one carrying the DOI."""
    dst = tmp_path / "with_doi.pdf"
    with ArticlePDF(NO_XMP_PDF) as pdf:
        assert pdf._xmp_packet() == ""
        pdf.set_doi(DOI, dst)

    with ArticlePDF(dst) as pdf:
        assert pdf._extract_doi_from_xmp() == DOI


def test_existing_doi_is_replaced(tmp_path: Path) -> None:
    """A DOI the PDF carries is replaced in place, not duplicated."""
    dst = tmp_path / "with_doi.pdf"
    with ArticlePDF(ELSEVIER_PDF) as pdf:
        assert pdf.doi == ELSEVIER_DOI
        pdf.set_doi(DOI, dst)

    with ArticlePDF(dst) as pdf:
        assert pdf.doi == DOI
        assert pdf._extract_doi_from_xmp() == DOI
        subject = pdf.file.metadata["subject"]
        assert subject.startswith("Applied Surface Science, 516 (2020) 145661")
        assert ELSEVIER_DOI not in subject
        xmp = pdf._xmp_packet()
        assert ELSEVIER_DOI not in xmp
        assert xmp.count("<prism:doi>") == 1
        assert xmp.count("<dc:identifier>") == 1


def test_in_place_save_is_incremental(tmp_path: Path) -> None:
    """Saving in place appends to the file and is read back on reopening."""
    path = _copy(NO_DOI_PDF, tmp_path)
    original = path.read_bytes()
    with ArticlePDF(path) as pdf:
        contents = _page_contents(pdf)
        pdf.set_doi(f"doi:{DOI}")
        assert pdf.doi == DOI

    assert path.read_bytes().startswith(original)
    with ArticlePDF(path) as pdf:
        assert pdf.doi == DOI
        assert _page_contents(pdf) == contents


def test_in_place_save_without_incremental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document that cannot be saved incrementally is rewritten in place."""
    path = _copy(NO_DOI_PDF, tmp_path)
    with ArticlePDF(path) as pdf:
        contents = _page_contents(pdf)
        monkeypatch.setattr(pdf.file, "can_save_incrementally", lambda: False)
        pdf.set_doi(DOI)
        assert pdf.doi == DOI
        assert _page_contents(pdf) == contents

    assert not path.with_name(f"{path.name}.tmp").exists()
    with ArticlePDF(path) as pdf:
        assert pdf.doi == DOI


@pytest.mark.parametrize("doi", ["", "not a doi", "10.1016", "https://example.org/x"])
def test_invalid_doi_raises(tmp_path: Path, doi: str) -> None:
    """An invalid DOI raises before anything is written."""
    path = _copy(NO_DOI_PDF, tmp_path)
    original = path.read_bytes()
    with ArticlePDF(path) as pdf:
        with pytest.raises(ValueError):
            pdf.set_doi(doi)
    assert path.read_bytes() == original


def test_pdf_from_bytes_needs_dst(tmp_path: Path) -> None:
    """A PDF opened from bytes has no file to save in place."""
    with ArticlePDF(NO_DOI_PDF.read_bytes()) as pdf:
        with pytest.raises(ValueError):
            pdf.set_doi(DOI)
        pdf.set_doi(DOI, tmp_path / "with_doi.pdf")

    with ArticlePDF(tmp_path / "with_doi.pdf") as pdf:
        assert pdf.doi == DOI
