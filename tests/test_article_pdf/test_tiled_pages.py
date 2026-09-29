"""
Tests for ArticlePDF pages drawn in tiles.
"""

from dataclasses import replace
from pathlib import Path

from pymupdf import Rect
from pytest import approx
from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import TextLinePDF, TextSpanPDF


# Get the test PDFs directory
TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"

# A SPIE proceedings paper whose fifth page was redrawn by a PDF compressor as a
# row of vertical strips about 53 pt wide. Every line of that page is cut into a
# fragment per strip, and a glyph crossing a seam is drawn in both strips.
TILED_PDF = (
    TEST_PDFS_DIR
    / "bare laser-synthesized si nanoparticles as functional elements for chitosan nanofiber-based tissue engineering platforms.pdf"
)
TILED_PAGE = 4
# An untiled article, whose lines include fragments split off at a superscript
# or a symbol font on the same baseline, but no repeated glyph.
UNTILED_PDF = TEST_PDFS_DIR / "laser ablation-based methods for nanostructuring of materials.pdf"


def _span(text: str, x0: float, x1: float, size: float = 10.0) -> TextSpanPDF:
    """Build a Times span on a baseline at y = 100."""

    return TextSpanPDF(
        rect=Rect(x0, 89.0, x1, 101.0),
        origin=(x0, 100.0),
        font="TimesNewRoman",
        ascender=0.9,
        descender=-0.2,
        size=size,
        flags=4,
        char_flags=16,
        color=(0.0, 0.0, 0.0),
        alpha=255,
        text=text,
    )


def _line(*spans: TextSpanPDF) -> TextLinePDF:
    """Build a horizontal line of spans."""

    rect = Rect()
    for span in spans:
        rect.include_rect(span.rect)
    return TextLinePDF(rect=rect, wmode=0, dir=(1.0, 0.0), spans=list(spans))


class TestJoinFragments:
    """Tests for joining the fragments of a line cut by a seam."""

    def test_repeated_glyph_is_kept_once(self) -> None:
        """Test that the glyph drawn on both sides of a seam appears once."""

        line = _line(_span("chitosan diss", 62.8, 116.0))
        fragment = _line(_span("solved in the", 112.1, 169.3))

        joined = ArticlePDF._join_fragments(line, fragment)

        assert joined.text == "chitosan dissolved in the"
        assert len(joined.spans) == 1
        rect = joined.rect
        assert (rect.x0, rect.y0, rect.x1, rect.y1) == approx((62.8, 89.0, 169.3, 101.0))

    def test_fragment_after_word_space_is_joined(self) -> None:
        """Test that a seam falling on a word space still continues the line."""

        line = _line(_span("cylindrical morphology", 274.3, 386.8))
        fragment = _line(_span("with mean d", 389.3, 441.4))

        assert ArticlePDF._is_tile_seam(line, fragment)
        assert not ArticlePDF._repeats_last_glyph(line, fragment)
        assert ArticlePDF._join_fragments(line, fragment).text == (
            "cylindrical morphology with mean d"
        )

    def test_fragment_of_the_next_column_is_not_a_seam(self) -> None:
        """Test that a line past a column gutter does not continue the line."""

        line = _line(_span("end of the left column", 60.0, 290.0))
        fragment = _line(_span("start of the right", 310.0, 400.0))

        assert not ArticlePDF._is_tile_seam(line, fragment)

    def test_different_font_stays_a_span_of_its_own(self) -> None:
        """Test that a fragment set in another font is appended as a new span."""

        line = _line(_span("the jet.", 62.8, 83.0))
        superscript = replace(_span("37,38", 83.7, 98.0, size=6.5), font="Symbol")

        joined = ArticlePDF._join_fragments(line, _line(superscript))

        assert [span.text for span in joined.spans] == ["the jet.", "37,38"]


class TestTiledPage:
    """Tests for a document with a page drawn in tiles."""

    def test_only_the_tiled_page_is_joined(self) -> None:
        """Test that every other page of the document keeps its blocks as drawn."""

        with ArticlePDF(TILED_PDF) as article:
            for page_no in range(article.file.page_count):
                blocks = article._raw_text_cache[page_no]
                widest = max(block.rect.width for block in blocks)
                if page_no == TILED_PAGE:
                    assert widest > article.paragraph_width.min
                else:
                    assert article._join_tile_fragments(blocks) is blocks

    def test_captions_are_whole(self) -> None:
        """Test that captions on the tiled page are read in full."""

        with ArticlePDF(TILED_PDF) as article:
            captions = article.figure_captions

        assert captions["2"] == (
            "Time-dependent size evolution of Si NPs, obtained under oxygen-rich "
            "and oxygen-free conditions."
        )
        assert captions["3"] == (
            "SEM micrographs of (a) pure chitosan(PEO) fibers and (b) hybrid "
            "chitosan(PEO) nanofibers functionalized with Si NPs"
        )

    def test_captions_are_not_side_captions(self) -> None:
        """Test that a caption cut to one strip no longer passes for a side caption."""

        with ArticlePDF(TILED_PDF) as article:
            figures = article.get_figures(TILED_PAGE)
            assert figures
            for figure in figures.values():
                assert figure.rect.y1 <= figure.caption.rect.y0

    def test_body_text_is_read_as_paragraphs(self) -> None:
        """Test that the body text of the tiled page is found as paragraphs."""

        with ArticlePDF(TILED_PDF) as article:
            paragraphs = article.get_paragraph_rects(TILED_PAGE)
            text = " ".join(block.text for block in article._text_cache[TILED_PAGE])

        assert sum(rect.width > 480 and rect.height > 50 for rect in paragraphs) == 2
        assert "chitosan dissolved in the range of 0.5–2 wt%" in text
        assert "disssolved" not in text

    def test_untiled_fragments_are_left_alone(self) -> None:
        """Test that fragments split off at a superscript are not taken for tiles."""

        with ArticlePDF(UNTILED_PDF) as article:
            for page_no in range(article.file.page_count):
                blocks = article._raw_text_cache[page_no]
                assert article._join_tile_fragments(blocks) is blocks
