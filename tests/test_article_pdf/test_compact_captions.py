"""Regression tests for captions with no space after the opening period."""

from pathlib import Path

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF


@pytest.mark.parametrize("opening", ["Fig.", "Figure.", "Supplementary Fig."])
def test_compact_captions_find_and_mark_figures(
    tmp_path: Path, opening: str
) -> None:
    """Find actual compact captions rather than spaced body references."""

    source = tmp_path / "compact.pdf"
    figure_rect = Rect(100, 200, 400, 380)
    with pymupdf.open() as document:
        for number in (1, 2):
            page = document.new_page()
            if number == 1:
                page.insert_text(
                    (72, 80),
                    "Figure S1 shows the measurements discussed in the text.",
                    fontname="tiro",
                    fontsize=12,
                )
            page.draw_rect(figure_rect, color=(0, 0, 1), fill=(0.8, 0.8, 1))
            page.insert_text(
                (100, 405),
                f"{opening}S{number} Measurements of sample {number}.",
                fontname="hebo",
                fontsize=10,
            )
        document.save(source)

    with ArticlePDF(source) as article:
        assert article.figure_captions == {
            f"S{number}": f"Measurements of sample {number}." for number in (1, 2)
        }
        assert set(article.figures) == {"S1", "S2"}
        for number in (1, 2):
            figure = article.figures[f"S{number}"]
            assert figure.rect == figure_rect
            assert article._figure_label_to_page_ind[f"S{number}"] == number - 1
        marked = article.mark("figures", output_path=tmp_path / "marked")

    with pymupdf.open(marked) as document:
        for page in document:
            marks = [
                drawing
                for drawing in page.get_drawings()
                if drawing["color"] == (1, 0, 0)
            ]
            assert len(marks) == 1
            assert marks[0]["rect"] == figure_rect
