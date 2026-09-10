"""
Tests for ArticlePDF header detection.
"""

import logging
import warnings

import pytest
from pathlib import Path
from artfinder.article_pdf import ArticlePDF


# Get the test PDFs directory
TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"

# A seven-page article whose only top-band content is the masthead and the
# "Check for updates" badge of its first page. The badge is drawn as two fills
# sharing one rectangle, which used to pass for a header repeated twice.
BADGE_ONLY_PDF = (
    TEST_PDFS_DIR
    / "tselikov-et-al-2022-transition-metal-dichalcogenide-nanospheres-for-high-refractive-index-nanophotonics-and-biomedical.pdf"
)
# A supplementary information file, with neither running header nor footer.
NO_FOOTER_PDF = (
    TEST_PDFS_DIR / "tunable nanostructuring for van der waals materials_si.pdf"
)
# An IOP article whose two-line running head ends below the 9% search band.
DEEP_HEADER_PDF = (
    TEST_PDFS_DIR
    / "comparison of pharmacokinetics and biodistribution of laser-synthesized plasmonic au and tin nanoparticles.pdf"
)
# A PNAS article with a rotated download stamp running down the page margin.
SIDEBAR_PDF = BADGE_ONLY_PDF
# An article whose pages end in captions and body text, with no running footer.
NO_FOOTER_CONTENT_PDF = (
    TEST_PDFS_DIR
    / "exciton photoluminescence from zno layers produced by laser-induced gas breakdown processing.pdf"
)
# A four-page article, the shortest layout that still carries a running header.
SHORT_ARTICLE_PDF = (
    TEST_PDFS_DIR / "cytotoxicity of laser-synthesized nanoparticles of elemental bismuth.pdf"
)
# A six-page article carrying both a running header and a running footer.
HEADER_AND_FOOTER_PDF = (
    TEST_PDFS_DIR
    / "laser-ablation synthesis of colloidal zrn nanoparticles in different liquids.pdf"
)


class TestHeaderThreshold:
    """Tests for the number of pages a header must appear on."""

    @pytest.mark.parametrize(
        ("page_count", "expected"),
        [(1, 2), (4, 2), (5, 2), (7, 3), (10, 4), (13, 6)],
    )
    def test_threshold_follows_page_count(
        self, page_count: int, expected: int
    ) -> None:
        """Test the threshold against the page count.

        Never below two, and never above the number of pages an alternating
        header would appear on.
        """

        assert max(2, (page_count - 1) // 2) == expected

    def test_short_article_keeps_its_header(self) -> None:
        """Test that a four-page article is not held to a higher threshold."""

        with ArticlePDF(SHORT_ARTICLE_PDF) as article:
            assert article.header_min_pages == 2
            assert article.header_rect.y1 > 0, "the running header was lost"


class TestHeaderPageCounting:
    """Tests that a header is counted by pages, not by occurrences."""

    def test_single_page_badge_is_not_a_header(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that stacked shapes on one page do not make a header."""

        with caplog.at_level(logging.DEBUG, logger="artfinder.article_pdf"):
            with ArticlePDF(BADGE_ONLY_PDF) as article:
                header = article.header_rect

        assert header.is_empty, f"a header was found where there is none: {header}"
        assert "Header rectangle not found" in caplog.text

    def test_missing_header_is_not_a_warning(self) -> None:
        """Test that a document without a header does not warn.

        Having no running header is normal for supplementary information and
        preprints, so it is reported at debug level only.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with ArticlePDF(BADGE_ONLY_PDF) as article:
                assert article.header_rect.is_empty

    def test_figures_are_not_clipped_by_a_missing_header(self) -> None:
        """Test that figures of a headerless document keep their top edge."""

        with ArticlePDF(BADGE_ONLY_PDF) as article:
            figures = article.figures

        # Figure 2 sits at the top of its page, above where the badge would have
        # put a header bound.
        assert figures["2"].rect.y0 < 40


class TestFooter:
    """Tests for footer detection."""

    def test_footer_found_below_the_content(self) -> None:
        """Test that a running footer is found at the bottom of the page."""

        with ArticlePDF(HEADER_AND_FOOTER_PDF) as article:
            footer = article.footer_rect
            page_rect = article.file[0].rect

        assert not footer.is_empty
        assert footer.y0 > page_rect.height * 0.8
        assert footer.y1 == page_rect.y1

    def test_missing_footer_is_not_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that a document without a footer reports it at debug level."""

        with caplog.at_level(logging.DEBUG, logger="artfinder.article_pdf"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with ArticlePDF(NO_FOOTER_PDF) as article:
                    assert article.footer_rect.is_empty

        assert "Footer rectangle not found" in caplog.text

    def test_header_and_footer_do_not_meet(self) -> None:
        """Test that the two bands stay at their own ends of the page."""

        with ArticlePDF(HEADER_AND_FOOTER_PDF) as article:
            header, footer = article.header_rect, article.footer_rect

        assert header.y1 < footer.y0


class TestBandCandidates:
    """Tests for what may take part in header and footer detection."""

    def test_header_covers_a_deep_running_head(self) -> None:
        """Test a header set deeper than the search band.

        This journal's header is two lines ending at 81 pt, past the 9% band of
        an A4 page; only its horizontal rule used to fit, which cut the header
        in half.
        """

        with ArticlePDF(DEEP_HEADER_PDF) as article:
            header = article.header_rect
            first_page_body = min(
                rect.y0 for rect in article.get_paragraph_rects(page_no=2)
            )

        assert header.y1 == pytest.approx(81.0, abs=1.0)
        assert header.y1 < first_page_body

    def test_sidebar_is_not_a_footer(self) -> None:
        """Test that an element running the height of the page is not furniture.

        This article carries a rotated "Downloaded from ..." stamp down its
        margin, repeating on most pages and reaching into the footer band.
        """

        with ArticlePDF(SIDEBAR_PDF) as article:
            footer = article.footer_rect
            page_height = article.file[0].rect.height

        assert not footer.is_empty
        assert footer.y0 > page_height * 0.9, "the sidebar dragged the footer up"

    def test_content_dipping_into_the_band_is_not_a_footer(self) -> None:
        """Test that a caption at the foot of a page does not make a footer."""

        with ArticlePDF(NO_FOOTER_CONTENT_PDF) as article:
            assert article.footer_rect.is_empty
