"""Clipping scopes and zero-area paths must not distort figure bounds."""

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF


@pytest.mark.parametrize(
    "hidden",
    [
        b"q 0 0 0 0 re W n 100 100 80 80 re f Q",
        # Both local clips have area; their intersection is empty.
        b"q 0 0 50 50 re W n q 100 100 50 50 re W n "
        b"100 100 80 80 re f Q Q",
        # Leaving an empty inner clip must restore its nonempty parent.
        b"q 0 0 400 400 re W n q 0 0 0 0 re W n "
        b"100 100 80 80 re f Q 20 20 10 10 re f Q",
    ],
    ids=["empty-clip", "disjoint-nested-clips", "restore-parent-clip"],
)
def test_empty_clip_hides_paths_without_leaking_to_siblings(hidden: bytes) -> None:
    """Discard only hidden paths, not the visible siblings following their scope."""
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=400)
        page.draw_rect(Rect(1, 1, 2, 2))
        stream = b"0 g " + hidden + b" 40 40 20 20 re f 40 60 m 60 60 l S"
        document.update_stream(page.get_contents()[0], stream)
        data = document.tobytes()
    with ArticlePDF(data, identifier="clipped-drawings") as article:
        drawings = article._drawings_cache[0]
        assert Rect(100, 220, 180, 300) not in [d.rect for d in drawings]
        assert len(drawings) == (3 if b"20 20 10 10" in hidden else 2)
        assert drawings[-2].rect == Rect(40, 340, 60, 360)
        # A visible horizontal stroke has an empty rectangle but must stay.
        assert drawings[-1].rect == Rect(40, 340, 60, 340)


def test_nonempty_clip_limits_drawing_bounds() -> None:
    """Intersect nested bounds, keep clipped hairlines, and restore the outer scope."""
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=400)
        page.draw_rect(Rect(1, 1, 2, 2))
        document.update_stream(
            page.get_contents()[0],
            b"q 80 80 160 160 re W n q 140 40 140 160 re W n "
            b"100 60 m 260 220 l S "  # A path extending past both clips.
            b"100 120 m 260 120 l S "  # A zero-height axis that crosses the clip.
            b"100 210 20 20 re f Q "  # Entirely outside the inner clip.
            b"100 210 20 20 re f Q "  # Same shape visible in the outer clip.
            b"250 250 20 20 re f",  # Unclipped sibling.
        )
        data = document.tobytes()
    with ArticlePDF(data, identifier="nonempty-clip") as article:
        assert article.get_drawing_rects(0) == [
            Rect(140, 200, 240, 320),
            Rect(140, 280, 240, 280),
            Rect(100, 170, 120, 190),
            Rect(250, 130, 270, 150),
        ]
        # The vector replay API still uses the unclipped path geometry.
        assert article._drawings_cache[0][0].rect == Rect(100, 180, 260, 340)


def test_plot_axes_are_included_in_figure_bounds() -> None:
    """Keep all four zero-area border paths, not just the data curve's area."""
    bounds = Rect(80, 100, 220, 240)
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=400)
        for start, end in [
            (bounds.tl, bounds.tr),
            (bounds.tr, bounds.br),
            (bounds.br, bounds.bl),
            (bounds.bl, bounds.tl),
        ]:
            page.draw_line(start, end, width=0.5)
        page.draw_rect(Rect(90, 110, 210, 230), color=(0, 0, 1))
        page.insert_text((80, 265), "Figure 1. Measurements.", fontsize=10)
        data = document.tobytes()
    with ArticlePDF(data, identifier="plot-axes") as article:
        assert article.figures["1"].rect == bounds


def test_an_isolated_hairline_is_not_a_figure_panel() -> None:
    """Preserving plot borders must not promote isolated strokes to panels."""
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=400)
        page.draw_line((80, 100), (220, 100), width=0.5)
        data = document.tobytes()
    with ArticlePDF(data, identifier="isolated-hairline") as article:
        assert article._figure_clusters_cache[0] == ()
