"""Text blocks grouping panel letters must not bridge separate figures."""

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF


@pytest.mark.parametrize("shared_caption", [False, True])
@pytest.mark.parametrize("label_rows", [1, 2])
def test_grouped_panel_letters(shared_caption: bool, label_rows: int) -> None:
    """Attach separate lines, then let captions decide figure membership."""
    panels = [Rect(60, 120, 240, 260), Rect(300, 120, 480, 260)]
    with pymupdf.open() as document:
        page = document.new_page()
        for panel in panels:
            page.draw_rect(panel, color=(0, 0, 1), fill=(0.7, 0.8, 1))
        for row in range(label_rows):
            for panel in panels:
                page.insert_text(
                    (panel.x0 + 5, 140 + 80 * row),
                    f"({chr(ord('a') + row)})",
                    fontname="hebo",
                    fontsize=10,
                )
        for index, panel in enumerate(panels[:1] if shared_caption else panels):
            page.insert_text(
                (panel.x0, 290 + 20 * index),
                f"Figure {index + 1}. Measurements.",
                fontname="hebo",
                fontsize=10,
            )
        data = document.tobytes()
    with ArticlePDF(data, identifier="grouped-labels") as article:
        # Verify the input really has the problematic extraction layout:
        # one text block per row, with a line in each separate graphic.
        grouped = [
            block
            for block in article._text_cache[0]
            if block.text in ("(a) (a)", "(b) (b)")
        ]
        assert len(grouped) == label_rows
        for block in grouped:
            assert len(block.lines) == 2
            assert all(block.rect.intersects(panel) for panel in panels)
        if shared_caption:
            assert set(article.figures) == {"1"}
            assert article.figures["1"].rect == panels[0] | panels[1]
        else:
            assert set(article.figures) == {"1", "2"}
            assert article.figures["1"].rect == panels[0]
            assert article.figures["2"].rect == panels[1]


def test_multiline_label_still_attaches_in_full() -> None:
    """Splitting the attachment units must not truncate a stacked axis title."""
    panel = Rect(100, 120, 280, 260)
    with pymupdf.open() as document:
        page = document.new_page()
        page.draw_rect(panel, color=(0, 0, 1), fill=(0.7, 0.8, 1))
        page.insert_text(
            (65, 180), "Mean\nsize", fontname="hebo", fontsize=10
        )
        page.insert_text(
            (100, 290), "Figure 1. Measurements.", fontname="hebo", fontsize=10
        )
        data = document.tobytes()
    with ArticlePDF(data, identifier="multiline-label") as article:
        label = next(block for block in article._text_cache[0] if block.text == "Mean size")
        assert len(label.lines) == 2
        assert article.figures["1"].rect == panel | label.rect
