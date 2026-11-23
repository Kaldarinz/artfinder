"""
Tests for reading an article's title from PDF metadata or its first page.
"""

from pathlib import Path

import pymupdf
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
        # No SI heading: the opening is followed by marked names and affiliations.
        (
            "10.1021_acsomega.9b01983_si.pdf",
            "Continuous flow copper laser ablation synthesis of copper (I and II) oxide"
            " nanoparticles in water",
        ),
        # A standalone SI heading in body type, then a double-spaced bold title.
        (
            "10.1021_acsomega.8b01606_si.pdf",
            "Laser ablated vortex fluidic mediated synthesis of superparamagnetic"
            " magnetite nanoparticles in water under flow",
        ),
        # The last line of the cover title starts just below the page midpoint.
        (
            "10.1021_acsomega.5c05967_si.pdf",
            "Effects of Au nanoparticles suspended in chlorobenzene antisolvent"
            " on mixed-halide perovskites",
        ),
        # Only the SI heading is enlarged; the two-line title above it is bold.
        (
            "10.1021_acsomega.5c04247_si.pdf",
            "Ag Nanoparticle Layer on PEDOT:PSS with Optimized Energy Levels"
            " for Improving PM6:Y6-Based Organic Photovoltaics",
        ),
    ],
)
def test_title(file_name: str, expected: str | None) -> None:
    """The title comes from metadata when it is printed on page 1, else from page 1."""
    with ArticlePDF(TEST_PDFS_DIR / file_name) as pdf:
        assert pdf.title == expected


def test_bold_table_labels_do_not_trigger_a_graphics_scan() -> None:
    """No SI heading or byline markers: reject this cover using text alone."""
    path = TEST_PDFS_DIR / (
        "silicon-gold nanoparticles affect wharton's jelly phenotype and"
        " secretome during tri-lineage differentiation_si.pdf"
    )
    with ArticlePDF(path) as pdf:
        assert pdf.title is None
        assert not pdf._drawings_cache
        assert not pdf._images_cache


_OPENING = ("A body-sized opening in bold", "hebo", 11)
_SI = ("Supplementary Information", "helv", 17)
_LARGE_TITLE = ("An ordinary enlarged article title", "helv", 18)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        pytest.param([_OPENING, _SI], _OPENING[0], id="bold-opening"),
        pytest.param([_OPENING], None, id="no-si-heading"),
        pytest.param([_OPENING, (_SI[0], "helv", 11)], None, id="body-sized-heading"),
        pytest.param([_OPENING, ("Supplementary Information: Contents", "helv", 17)],
                     None, id="heading-not-standalone"),
        pytest.param([_SI, _OPENING], None, id="bold-text-below-heading"),
        pytest.param([(_OPENING[0], "helv", 11), _SI], None, id="not-bold"),
        pytest.param([(_OPENING[0], "hebo", 8), _SI], None, id="smaller-than-body"),
        pytest.param([("Cover note", "helv", 11), _OPENING, _SI], None, id="not-opening"),
        pytest.param([_OPENING, _SI, _LARGE_TITLE], _LARGE_TITLE[0], id="ordinary-title-wins"),
    ],
)
def test_bold_si_title_fallback(
    lines: list[tuple[str, str, int]], expected: str | None
) -> None:
    """Only a bold, body-sized opening above a prominent SI heading is a fallback."""
    with pymupdf.open() as doc:
        page = doc.new_page()
        for i, (text, font, size) in enumerate(lines):
            page.insert_text((60, 60 + 40 * i), text, fontname=font, fontsize=size)
        for i in range(12):
            page.insert_text((60, 400 + 16 * i), "Ordinary body text.", fontsize=11)
        with ArticlePDF(doc.tobytes(), identifier="si-cover") as pdf:
            assert pdf.title == expected


@pytest.mark.parametrize(("byline_font", "byline_y"), [
    ("tibo", 116),  # still bold and consecutive, but a different family
    ("hebi", 116),  # same family, different emphasis
    ("hebo", 145),  # same style, no longer at title spacing
])
def test_heading_first_title_stops_at_byline(byline_font: str, byline_y: int) -> None:
    """A small size variation continues a title; new styling or a blank line ends it."""
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 60), "Supporting Information", fontsize=11)
        page.insert_text((72, 80), "A bold title with", fontname="hebo", fontsize=11)
        page.insert_text((72, 98), "a second line", fontname="hebo", fontsize=11.2)
        page.insert_text((72, byline_y), "First Author and Second Author", fontname=byline_font, fontsize=11)
        for i in range(12):
            page.insert_text((72, 300 + 14 * i), "Ordinary text of the article.", fontsize=11)
        with ArticlePDF(doc.tobytes(), identifier="si-cover") as pdf:
            assert pdf.title == "A bold title with a second line"


@pytest.mark.parametrize(("second_name", "affiliation", "byline_y", "expected"), [
    ("Sam Lee", "1", 96, _OPENING[0]),
    ("Sam Lee", "3", 96, None),           # unrelated footnote
    ("another sample", "1", 96, None),    # prose with exponents, not two marked names
    ("Sam Lee", "1", 200, None),          # byline too far from the opening
])
def test_headingless_title_requires_a_marked_byline(
    second_name: str, affiliation: str, byline_y: int, expected: str | None
) -> None:
    """Use the superscript name markers only when an affiliation corroborates them."""
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 76), _OPENING[0], fontsize=11, fontname="hebo")
        x = 72.0
        for text, size, rise in [("Jane Smith", 11, 0), ("1", 7, 4),
                                 (f", {second_name}", 11, 0), ("2", 7, 4)]:
            page.insert_text((x, byline_y - rise), text, fontsize=size)
            x += pymupdf.get_text_length(text, fontsize=size)
        page.insert_text((72, byline_y + 14), affiliation, fontsize=7)
        page.insert_text((78, byline_y + 17), "Department of Chemistry", fontsize=10)
        for i in range(12):
            page.insert_text((72, 300 + i * 14), "Ordinary body text.", fontsize=11)
        with ArticlePDF(doc.tobytes(), identifier="cover") as pdf:
            assert pdf.title == expected
            assert not pdf.is_supplement


def test_title_cannot_start_below_midpage() -> None:
    """A separate same-font group below midpage must not compete with the title."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=800)
        page.insert_text((60, 360), "The real cover title", fontsize=18)
        page.insert_text((60, 490), "A much longer heading in the same font", fontsize=18)
        for i in range(12):
            page.insert_text((60, 540 + 14 * i), "Ordinary body text.", fontsize=10)
        with ArticlePDF(doc.tobytes(), identifier="cover") as pdf:
            assert pdf.title == "The real cover title"


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
