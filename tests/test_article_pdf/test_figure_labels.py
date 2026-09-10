"""
Tests for ArticlePDF figure labels.
"""

import re
from statistics import median

import pytest
from pathlib import Path
from pymupdf import Rect
from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import TextLinePDF


# Get the test PDFs directory
TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"
SUPPLEMENTARY_PDF = (
    TEST_PDFS_DIR
    / "laser fragmentation of colloidal gold nanoparticles with high-intensity nanosecond pulses_si.pdf"
)
LINE_NUMBERED_PDF = (
    TEST_PDFS_DIR / "tunable nanostructuring for van der waals materials_si.pdf"
)


def _numbered_line(page_no: int) -> TextLinePDF:
    """Build a one-line stub at a fixed left margin, one per page."""

    return TextLinePDF(
        rect=Rect(55.0, 100.0, 70.0, 113.0), wmode=0, dir=(1.0, 0.0), spans=[]
    )


class TestCaptionLabel:
    """Tests for the label derived from a caption's opening words."""

    @pytest.mark.parametrize(
        ("caption", "expected"),
        [
            ("Figure 1. Sketch of the setup.", "1"),
            ("Fig. 1 Sketch of the setup.", "1"),
            ("Fig 1 Sketch of the setup.", "1"),
            ("Figure S1. Extinction spectra.", "S1"),
            ("Fig. S12: Size distributions.", "S12"),
            ("Figure S 1 Relative mass distribution.", "S1"),
            ("FIGURE S4. TEM images.", "S4"),
            ("Figure S01. Leading zeros are dropped.", "S1"),
            ("Supplementary Figure 3 Cell viability.", "S3"),
            ("Supplementary Fig. S3 Cell viability.", "S3"),
            ("Supplemental Figure 2. Photographs of solutions.", "S2"),
            ("Supporting Figure 2. Photographs of solutions.", "S2"),
        ],
    )
    def test_label_of_caption(self, caption: str, expected: str) -> None:
        """Test that a caption's first line yields the expected figure label."""

        match = ArticlePDF.CAPTION_PATTERN.match(caption)
        assert match is not None, f"Caption not recognized: {caption!r}"
        assert ArticlePDF._caption_label(match) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Figures 1 and 2 show the setup.",
            "Fig. SI1 is not supported yet.",
            "Figure S-1 is not supported yet.",
            "Figure IV is not supported yet.",
            "The figure below shows the setup.",
        ],
    )
    def test_non_caption_is_not_matched(self, text: str) -> None:
        """Test that text without a supported figure label is not a caption."""

        assert ArticlePDF.CAPTION_PATTERN.match(text) is None


class TestSupplementaryFigures:
    """Tests for figures of a supplementary information document."""

    def test_supplementary_labels(self) -> None:
        """Test that supplementary captions are labelled with an `S` prefix."""

        with ArticlePDF(SUPPLEMENTARY_PDF) as article:
            labels = set(article.figures.keys())

        assert labels == {f"S{i}" for i in range(1, 12)}

    def test_extract_figures_keyed_by_label(self, tmp_path: Path) -> None:
        """Test that `extract_figures` returns a file per figure, keyed by label."""

        with ArticlePDF(SUPPLEMENTARY_PDF) as article:
            paths = article.extract_figures(output_path=tmp_path)
            labels = set(article.figures.keys())

        assert set(paths.keys()) == labels
        for label, path in paths.items():
            assert path.exists(), f"Figure {label} was not written to {path}"
            assert path.name.endswith(f"_fig_{label}.png"), path.name


class TestLineNumbers:
    """Tests for manuscript line number detection."""

    def test_line_numbers_found_in_numbered_manuscript(self) -> None:
        """Test that every line number of a numbered manuscript is found."""

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            found = article._line_number_rects

        assert sum(len(rects) for rects in found.values()) == 92

    def test_line_numbers_removed_from_text_blocks(self) -> None:
        """Test that no detected line number survives in the text blocks."""

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            for page_no in range(article.file.page_count):
                line_numbers = article._line_number_rects[page_no]
                for block in article._text_cache[page_no]:
                    for line in block.lines:
                        assert article._rect_key(line.rect) not in line_numbers, (
                            f"line number {line.text!r} survived on page {page_no}"
                        )

    def test_caption_block_loses_its_line_number(self) -> None:
        """Test that a line number merged into a caption block is dropped.

        The caption of figure S1 shares its block with the line number `29`,
        which used to end up in the caption text.
        """

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            caption = article.figure_captions["S1"]

        assert not caption.endswith("29")
        assert "pictures" in caption

    def test_numbered_list_is_kept(self) -> None:
        """Test that a numbered list is not mistaken for line numbering.

        Page 7 of the fixture carries a list numbered 1-6, 1-5, 1-4, ...; a
        restart is what tells it apart from line numbering.
        """

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            kept = [
                line.text.strip()
                for block in article._text_cache[7]
                for line in block.lines
                if re.fullmatch(r"\d{1,4}", line.text.strip())
            ]

        assert len(kept) == 31

    def test_no_line_numbers_in_published_articles(self) -> None:
        """Test that no line numbers are detected in a typeset article."""

        for name in (
            "laser-ablation synthesis of colloidal zrn nanoparticles in different liquids.pdf",
            "laser fragmentation of colloidal gold nanoparticles with high-intensity nanosecond pulses_si.pdf",
        ):
            with ArticlePDF(TEST_PDFS_DIR / name) as article:
                found = article._line_number_rects
            assert sum(len(rects) for rects in found.values()) == 0, name

    def test_page_numbers_are_not_line_numbers(self) -> None:
        """Test that a page number, being alone on its page, is not a column."""

        column = [
            (page_no, 100 + page_no, _numbered_line(page_no))
            for page_no in range(20)
        ]
        assert not ArticlePDF._is_line_number_column(column, page_count=20)

    def test_gutter_excludes_the_line_number_column(self) -> None:
        """Test that the searchable area of a page starts past the line numbers."""

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            left, right = article._line_number_gutter(4)
            column = max(rect[2] for rect in article._line_number_rects[4])
            page_rect = article.file[4].rect

        assert left > column, "the gutter still includes the line number column"
        assert right == page_rect.x1, "nothing numbers the right margin here"

    def test_gutter_is_the_full_page_without_line_numbers(self) -> None:
        """Test that a page without line numbers keeps the full search area."""

        with ArticlePDF(SUPPLEMENTARY_PDF) as article:
            page_rect = article.file[1].rect
            assert article._line_number_gutter(1) == (page_rect.x0, page_rect.x1)

    def test_figures_exclude_the_line_number_column(self) -> None:
        """Test that no figure rectangle reaches into the line number column."""

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            for label, figure in article.figures.items():
                page_no = article._figure_label_to_page_ind[label]
                column = [rect[2] for rect in article._line_number_rects[page_no]]
                assert figure.rect.x0 > max(column), (
                    f"figure {label} reaches into the line number column"
                )


class TestLinePitch:
    """Tests for the measured line pitch and the caption stitching using it."""

    def test_pitch_exceeds_line_height_in_a_manuscript(self) -> None:
        """Test that extra leading is measured rather than assumed away."""

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            pitch = article.line_pitch
            heights = [
                line.rect.height
                for block in article._text_cache[4]
                for line in block.lines
            ]

        assert pitch == pytest.approx(15.84, abs=0.1)
        assert pitch > max(heights) + 1, "the leading of this document is not extra"

    def test_pitch_matches_the_line_numbers(self) -> None:
        """Test the measured pitch against the line numbers of the same document.

        Line numbering marks every text line, so the distance between
        consecutive numbers is an independent measure of the pitch.
        """

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            pitch = article.line_pitch
            gaps = []
            for rects in article._line_number_rects.values():
                tops = sorted(rect[1] for rect in rects)
                gaps += [b - a for a, b in zip(tops, tops[1:]) if b - a < 100]

        assert median(gaps) == pytest.approx(pitch, abs=0.5)

    def test_caption_is_stitched_across_blocks(self) -> None:
        """Test that a caption spread over several blocks is kept whole."""

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            caption = article.figure_captions["S1"]

        assert caption.startswith("Three-dimensional representation")
        assert caption.endswith("M(A)Xenes, Others).")


class TestParagraphWidth:
    """Tests for the paragraph width used to bound figures."""

    def test_table_does_not_win_over_body_text(self) -> None:
        """Test that a tall table is not taken for the body paragraph width.

        Page 7 of the fixture holds a table of d-spacings as six blocks of
        16-24 short lines, 261 pt wide, whose combined height exceeds that of
        the body text of the whole document.
        """

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            width = article.paragraph_width
            columns = article.columns_number

        assert width.mean == pytest.approx(468.3, abs=1.0)
        assert columns == 1, "a single-column document was read as two columns"

    def test_body_paragraphs_are_recognised(self) -> None:
        """Test that the body paragraphs of a page match the paragraph width."""

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            paragraphs = article.get_paragraph_rects(8)

        assert len(paragraphs) > 5, "no body paragraph matched the paragraph width"

    def test_figure_excludes_the_paragraph_above_it(self) -> None:
        """Test that a figure is bounded by the paragraph above it.

        Figure S5 has three lines of body text above it, each its own block in
        this document. While the paragraph width was the width of a table, none
        of them bounded the figure and all three were swallowed.
        """

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            figure = article.figures["S5"]
            full_width_lines = [
                block.rect
                for block in article._text_cache[8]
                if block.rect.y1 < 110
                and block.rect.width >= article.paragraph_width.min
            ]

        assert len(full_width_lines) == 2, "the fixture changed"
        for rect in full_width_lines:
            assert not figure.rect.contains(rect), (
                f"figure S5 swallowed the text block at {tuple(rect)}"
            )

    def test_figure_excludes_the_last_line_of_the_paragraph_above(self) -> None:
        """Test that a short last line above a figure is not absorbed into it.

        The last line of a paragraph is narrower than the body, so it is not a
        paragraph in its own right; it bounds the figure only once it has been
        joined to the paragraph it belongs to.
        """

        with ArticlePDF(LINE_NUMBERED_PDF) as article:
            figure = article.figures["S5"]
            image = article._images_cache[8][0].rect

        assert figure.rect.y0 >= image.y0
        assert tuple(figure.rect) == pytest.approx(tuple(image))

    def test_hanging_indent_lines_join_their_paragraph(self) -> None:
        """Test that the continuation lines of a reference list are paragraphs.

        A hanging indent makes every line after the first narrower than the
        body, so none of them has the paragraph width on its own.
        """

        pdf = (
            TEST_PDFS_DIR
            / "cytotoxicity of laser-synthesized nanoparticles of elemental bismuth.pdf"
        )
        with ArticlePDF(pdf) as article:
            paragraphs = article.get_paragraph_rects(3)
            trailing = [
                block.rect
                for block in article._text_cache[3]
                if block.rect.y0 > 615 and block.rect.width < article.paragraph_width.min
            ]

        assert trailing, "the fixture no longer has narrow trailing lines"
        for rect in trailing:
            assert any(
                paragraph.contains(rect) for paragraph in paragraphs
            ), f"the trailing line at {tuple(rect)} belongs to no paragraph"


class TestSupplementaryFixtures:
    """Tests for the supplementary information documents of the golden set."""

    def test_combined_document_keeps_the_main_figures(self) -> None:
        """Test an article with its supplementary information bound to it.

        The four main captions of this article are set in one font and its three
        supplementary ones in another; the caption filter keeps whichever font is
        the more common in the document, so the supplementary set is lost. This
        records that limit rather than endorsing it.
        """

        pdf = (
            TEST_PDFS_DIR
            / "whole-cell patch-clamp measurements of spermatozoa reveal an alkaline-activated ca channel_with_si.pdf"
        )
        with ArticlePDF(pdf) as article:
            labels = set(article.figures)
            supplementary_captions = [
                block.text.strip()
                for page_no in range(article.file.page_count)
                for block in article._text_cache[page_no]
                if block.text.strip().lower().startswith("supplementary figure")
            ]

        assert labels == {"1", "2", "3", "4"}
        assert len(supplementary_captions) == 3, "the fixture changed"

    def test_supplementary_file_without_body_text(self) -> None:
        """Test a supplementary file made of captions and figures only.

        Its handful of single-line captions cluster nowhere, so the paragraph
        width cannot be measured; the document is then read as single-column.
        """

        pdf = (
            TEST_PDFS_DIR
            / "boron nanoparticle-enhanced proton therapy for cancer treatment_si.pdf"
        )
        with ArticlePDF(pdf) as article:
            assert article.paragraph_width.mean == 0.0
            assert article.columns_number == 1
            assert set(article.figures) == {"S1", "S2", "S3", "S4", "S5"}

    def test_two_column_supplementary_file(self) -> None:
        """Test a two-column supplementary file whose pages hold no paragraph.

        A figure in a column with no body text beside it used to raise, as the
        column boundary was taken from an empty list of paragraphs.
        """

        pdf = (
            TEST_PDFS_DIR
            / "laser-synthesized plasmonic hfn-based nanoparticles as a novel multifunctional agent for photothermal therapy_si.pdf"
        )
        with ArticlePDF(pdf) as article:
            assert article.columns_number == 2
            assert set(article.figures) == {f"S{i}" for i in range(1, 8)}
