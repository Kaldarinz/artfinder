"""Scale-aware grouping of widely spaced panels without merging nearby content."""

from collections.abc import Iterator

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF


@pytest.fixture
def article() -> Iterator[ArticlePDF]:
    """Provide a minimal document for panel-spacing geometry checks.

    Returns
    -------
    Iterator[ArticlePDF]
        Open one-page article.
    """
    with pymupdf.open() as document:
        document.new_page()
        data = document.tobytes()
    with ArticlePDF(data, identifier="panel-spacing") as pdf:
        yield pdf


@pytest.mark.parametrize(
    ("other", "caption", "expected"),
    [
        (Rect(300, 120, 480, 260), Rect(60, 280, 480, 300), True),
        (Rect(300, 120, 480, 260), Rect(60, 80, 480, 100), True),
        (Rect(300, 120, 480, 260), Rect(60, 180, 240, 200), False),
        (Rect(300, 120, 330, 260), Rect(60, 280, 480, 300), False),
        (Rect(340, 120, 520, 260), Rect(60, 280, 520, 300), False),
        (Rect(300, 200, 480, 340), Rect(60, 360, 480, 380), False),
        (Rect(60, 320, 240, 460), Rect(60, 480, 480, 500), False),
        (Rect(270, 290, 450, 430), Rect(60, 450, 480, 470), True),
    ],
    ids=[
        "wide-row-above-caption",
        "wide-row-below-caption",
        "caption-between-panels",
        "small-neighbour",
        "too-far-even-for-large-panels",
        "diagonally-offset",
        "vertical-gap-still-absolute",
        "original-diagonal-gap-preserved",
    ],
)
def test_panel_gap_geometry(
    article: ArticlePDF, other: Rect, caption: Rect, expected: bool
) -> None:
    """Extend only the horizontal gap for substantial, aligned panels."""
    rect = Rect(60, 120, 240, 260)
    assert article._within_figure_panel_gap(rect, other, caption) is expected
    assert article._within_figure_panel_gap(other, rect, caption) is expected


@pytest.mark.parametrize("caption_x", [60, 300])
@pytest.mark.parametrize("caption_y", [100, 290])
def test_wide_panels_share_one_caption(caption_x: int, caption_y: int) -> None:
    """Grow from either initial panel, with the caption above or below."""
    left, right = Rect(60, 120, 240, 260), Rect(300, 120, 480, 260)
    with pymupdf.open() as document:
        page = document.new_page()
        for rect in (left, right):
            page.draw_rect(rect, color=(0, 0, 1), fill=(0.7, 0.8, 1))
        page.insert_text(
            (caption_x, caption_y), "Figure 1. Measurements.", fontname="hebo", fontsize=10
        )
        data = document.tobytes()
    with ArticlePDF(data, identifier="panel-spacing") as pdf:
        assert set(pdf.figures) == {"1"}
        assert pdf.figures["1"].rect == left | right


def test_wide_panels_with_separate_captions_stay_separate() -> None:
    """Never steal a distant panel assigned to another figure's caption."""
    panels = [Rect(60, 120, 240, 260), Rect(300, 120, 480, 260)]
    with pymupdf.open() as document:
        page = document.new_page()
        for number, rect in enumerate(panels, start=1):
            page.draw_rect(rect, color=(0, 0, 1), fill=(0.7, 0.8, 1))
            page.insert_text(
                (rect.x0, 290 + 20 * (number - 1)),
                f"Figure {number}. Measurements.",
                fontname="hebo",
                fontsize=10,
            )
        data = document.tobytes()
    with ArticlePDF(data, identifier="panel-spacing") as pdf:
        assert set(pdf.figures) == {"1", "2"}
        assert pdf.figures["1"].rect == panels[0]
        assert pdf.figures["2"].rect == panels[1]


def test_paragraph_between_wide_panels_blocks_growth() -> None:
    """The scale-aware allowance does not bypass text obstacles."""
    left, right = Rect(40, 120, 240, 300), Rect(330, 120, 530, 300)
    with pymupdf.open() as document:
        page = document.new_page()
        for rect in (left, right):
            page.draw_rect(rect, color=(0, 0, 1), fill=(0.7, 0.8, 1))
        page.insert_text(
            (40, 330), "Figure 1. Measurements.", fontname="hebo", fontsize=10
        )
        page.insert_text(
            (245, 200),
            "This paragraph separates the two independent plots.",
            fontsize=3,
        )
        data = document.tobytes()
    with ArticlePDF(data, identifier="panel-spacing") as pdf:
        assert pdf.get_paragraph_rects(0)
        assert pdf.figures["1"].rect == left
