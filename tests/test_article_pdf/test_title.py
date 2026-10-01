"""
Tests for reading an article's title from PDF metadata or its first page.
"""

from pathlib import Path

import pytest

from artfinder.article_pdf import ArticlePDF

TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        # The metadata title is the title.
        (
            "plasmonic nanorod metamaterials for biosensing.pdf",
            "Plasmonic nanorod metamaterials for biosensing",
        ),
        (
            "organic solvent and surfactant free fluorescent organic nanoparticles"
            " by laser ablation of aggregation-induced enhanced emission dyes.pdf",
            "Organic Solvent and Surfactant Free Fluorescent Organic Nanoparticles"
            " by Laser Ablation of Aggregation‐Induced Enhanced Emission Dyes",
        ),
        # The metadata title is a file name: the first page is read.
        (
            "binary proton therapy of ehrlich carcinoma using targeted gold"
            " nanoparticles.pdf",
            "Binary Proton Therapy of Ehrlich Carcinoma Using Targeted Gold"
            " Nanoparticles",
        ),
        # "untitled".
        (
            "radio frequency radiation-induced hyperthermia using si"
            " nanoparticle-based sensitizers for mild cancer therapy.pdf",
            "Radio frequency radiation-induced hyperthermia using Si"
            " nanoparticle-based sensitizers for mild cancer therapy",
        ),
        # The metadata title belongs to another article.
        (
            "ex vivo biodistribution of gallium-68-labeled porous silicon"
            " nanoparticles.pdf",
            "Ex vivo biodistribution of gallium-68-labeled porous silicon"
            " nanoparticles",
        ),
        # No metadata title; a cover page sets the proceedings name larger still.
        (
            "bare laser-synthesized si nanoparticles as functional elements for"
            " chitosan nanofiber-based tissue engineering platforms.pdf",
            "Bare laser-synthesized Si nanoparticles as functional elements for"
            " chitosan nanofiber-based tissue engineering platforms",
        ),
        # Supporting information: the heading above the title is dropped, and the
        # author list in the title's font, one gap further down, stays out.
        (
            "organic solvent and surfactant free fluorescent organic nanoparticles"
            " by laser ablation of aggregation-induced enhanced emission dyes_si.pdf",
            "Organic Solvent and Surfactant Free Fluorescent Organic Nanoparticles"
            " by Laser Ablation of Aggregation-Induced Enhanced Emission Dyes",
        ),
        # A title wrapped at the hyphen of a compound keeps the hyphen.
        (
            "transition metal dichalcogenide nanospheres for high-refractive-index"
            " nanophotonics and biomedical theranostics_si.pdf",
            "Transition metal dichalcogenide nanospheres for high-refractive-index"
            " nanophotonics and biomedical theranostics",
        ),
        # Nothing on the first page is set larger than the body.
        (
            "silicon-gold nanoparticles affect wharton's jelly phenotype and"
            " secretome during tri-lineage differentiation_si.pdf",
            None,
        ),
    ],
)
def test_title(file_name: str, expected: str | None) -> None:
    """The title comes from metadata when it is printed on page 1, else from page 1."""
    with ArticlePDF(TEST_PDFS_DIR / file_name) as pdf:
        assert pdf.title == expected


def test_journal_banner_is_not_the_title() -> None:
    """An Elsevier banner set a little larger than the title loses to it."""
    file_name = (
        "localized infrared radiation-induced hyperthermia sensitized by"
        " laser-ablated silicon nanoparticles for phototherapy applications.pdf"
    )
    with ArticlePDF(TEST_PDFS_DIR / file_name) as pdf:
        assert pdf._title_from_first_page() == (
            "Localized infrared radiation-induced hyperthermia sensitized by"
            " laser-ablated silicon nanoparticles for phototherapy applications"
        )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("", None),
        ("untitled", None),
        ("Unbekannt", None),
        ("Microsoft Word - Manuscript final.docx", None),
        ("BullLeb2322015Skribitskaya.fm", None),
        ("1-s2.0-S0169433225012620-mmc1.docx", None),
        ("DOI: 10", None),
        ("Supporting Online Material for", None),
        ("Three word title", None),
        (
            "Aggregation&#x02010;Induced  Enhanced\nEmission Dyes",
            "Aggregation‐Induced Enhanced Emission Dyes",
        ),
        (
            "Supplementary Materials: Laser-Ablative Synthesis of Samarium Oxide",
            "Laser-Ablative Synthesis of Samarium Oxide",
        ),
    ],
)
def test_clean_metadata_title(title: str, expected: str | None) -> None:
    """Placeholders, file names and DOIs are not titles; entities are decoded."""
    assert ArticlePDF._clean_metadata_title(title) == expected
