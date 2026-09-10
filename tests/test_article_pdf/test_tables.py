"""
Tests for ArticlePDF table detection.
"""

import pytest
from pathlib import Path
from artfinder.article_pdf import ArticlePDF


# Get the test PDFs directory
TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"

# Two tables printed below their captions, in a supplementary information file.
CAPTION_ABOVE_PDF = (
    TEST_PDFS_DIR / "tunable nanostructuring for van der waals materials_si.pdf"
)
# Two tables printed above their captions, in a Nature-style article.
CAPTION_BELOW_PDF = (
    TEST_PDFS_DIR
    / "in vivo evaluation of safety, biodistribution and pharmacokinetics of laser-synthesized gold nanoparticles.pdf"
)
# A table followed by the page footer of the journal.
FOOTER_AFTER_TABLE_PDF = (
    TEST_PDFS_DIR
    / "effect of oxygen on colloidal stability of titanium nitride nanoparticles synthesized by laser ablation in liquids.pdf"
)


class TestTableCaptions:
    """Tests for finding tables through their captions."""

    def test_supplementary_table_labels(self) -> None:
        """Test that supplementary tables are labelled like supplementary figures."""

        with ArticlePDF(CAPTION_ABOVE_PDF) as article:
            labels = set(article.tables.keys())

        assert labels == {"S1", "S2"}

    def test_caption_text_is_kept_whole(self) -> None:
        """Test that a table caption is stitched across blocks like a figure's."""

        with ArticlePDF(CAPTION_ABOVE_PDF) as article:
            caption = article.table_captions["S2"]

        assert caption.startswith("Bending moduli")
        assert caption.endswith("sources.")

    def test_document_without_tables(self) -> None:
        """Test that an article with no table reports none."""

        with ArticlePDF(
            TEST_PDFS_DIR
            / "laser-ablation synthesis of colloidal zrn nanoparticles in different liquids.pdf"
        ) as article:
            assert article.tables == {}


class TestTableSide:
    """Tests for deciding which side of its caption a table is printed on."""

    def test_caption_above_the_body(self) -> None:
        """Test a table printed below its caption."""

        with ArticlePDF(CAPTION_ABOVE_PDF) as article:
            for label in ("S1", "S2"):
                table = article.tables[label]
                assert table.caption_above, f"table {label}"
                assert table.rect.y0 >= table.caption.rect.y1

    def test_caption_below_the_body(self) -> None:
        """Test a table printed above its caption."""

        with ArticlePDF(CAPTION_BELOW_PDF) as article:
            for label in ("1", "2"):
                table = article.tables[label]
                assert not table.caption_above, f"table {label}"
                assert table.rect.y1 <= table.caption.rect.y0


class TestTableExtent:
    """Tests for where a table body ends."""

    def test_body_stops_before_the_references(self) -> None:
        """Test that the text following a table is not taken for its rows."""

        with ArticlePDF(CAPTION_ABOVE_PDF) as article:
            table = article.tables["S2"]
            references = [
                block.rect
                for block in article._text_cache[8]
                if block.text.strip().startswith("References")
            ]

        assert references, "the fixture no longer has a reference list"
        assert table.rect.y1 < references[0].y0

    def test_body_stops_before_the_footer(self) -> None:
        """Test that the running footer is not taken for the last row."""

        with ArticlePDF(FOOTER_AFTER_TABLE_PDF) as article:
            table = article.tables["1"]
            footer = article.footer_rect

        assert not footer.is_empty, "the fixture no longer has a footer"
        assert table.rect.y1 < footer.y0

    def test_body_stays_in_the_column_of_its_caption(self) -> None:
        """Test that a table of a two-column page does not span both columns."""

        with ArticlePDF(
            TEST_PDFS_DIR
            / "nanocomposites composed of p3ht_pcbm and nanoparticles synthesized by laser ablation of a bulk pbs target in liquid.pdf"
        ) as article:
            table = article.tables["1"]
            page_width = article.file[5].rect.width

        assert table.rect.width < page_width * 0.6, "the table spans both columns"
        assert table.rect.x0 > page_width * 0.4, "the table is not in the right column"


class TestTablesAndParagraphWidth:
    """Tests that table rows do not pass for body paragraphs."""

    def test_table_rows_do_not_set_the_paragraph_width(self) -> None:
        """Test that the paragraph width is that of the running text.

        The supplementary information holds a table whose rows carry more total
        height than the body text of the document.
        """

        with ArticlePDF(CAPTION_ABOVE_PDF) as article:
            width = article.paragraph_width
            table = article.tables["S1"]

        assert width.mean == pytest.approx(468.3, abs=1.0)
        assert table.rect.width < width.min, "the table is as wide as a paragraph"
