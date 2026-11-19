"""An explicit caption marker can point across the body text's column gutter."""

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF


@pytest.mark.parametrize(
    ("caption_on_left", "marker", "marker_y", "blocked", "matches"),
    [
        (True, "→", 80, False, True),
        (False, "←", 80, False, True),
        (True, "", 80, False, False),
        (True, "←", 80, False, False),
        (True, "→", 95, False, False),
        (True, "→", 80, True, False),
    ],
    ids=["right", "left", "no-marker", "wrong-direction", "wrong-row", "blocked"],
)
def test_marked_side_caption_across_gutter(
    caption_on_left: bool, marker: str, marker_y: int, blocked: bool, matches: bool
) -> None:
    """Follow a marker in the opening row, not an unmarked or obstructed pair."""
    caption_x, figure_x = (40, 320) if caption_on_left else (320, 40)
    figure = Rect(figure_x, 70, figure_x + 236, 370)
    opening = "Figure 1. Spectra of irradiated and untreated samples."
    prose = "Results from selected regions were accumulated over time.\n" * 3
    with pymupdf.open() as document:
        document.set_metadata({"subject": "doi:10.1234/directional-side-caption"})
        page = document.new_page(width=600, height=700)
        page.draw_rect(figure, color=(0, 0, 1), fill=(0.8, 0.8, 1))
        page.insert_text(
            (caption_x, 80),
            f"{opening}\nIrradiated samples are shown in blue.\n"
            "The control is shown in grey.",
            fontname="hebo",
            fontsize=9,
        )
        if marker:
            opening_width = pymupdf.get_text_length(opening, fontname="hebo", fontsize=9)
            marker_x = caption_x + opening_width + 2 if caption_on_left else caption_x - 11
            page.insert_text((marker_x, marker_y), marker, fontname="china-s", fontsize=9)
        if blocked:
            page.insert_text((284, 102), "Table 1. Data", fontsize=5)
        # A paragraph under the caption is inside the pair's enclosing box,
        # but is not between the two. Repeat the body width in both columns.
        for x, ys in [(caption_x, (180, 250, 320)), (figure_x, (420, 490, 560))]:
            for y in ys:
                page.insert_text((x, y), prose, fontsize=9)
        pdf = document.tobytes()

    with ArticlePDF(pdf) as article:
        assert article._column_gutter_middle(0) is not None
        if blocked:
            assert len(article.get_table_caption_rects(0)) == 1
        assert (article.figures["1"].rect == figure) is matches
