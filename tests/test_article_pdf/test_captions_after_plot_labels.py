"""Captions grouped with preceding plot labels remain separate from graphics."""

from dataclasses import replace

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import TextBlockPDF


def _plot_label_pdf(opening: str, label_lines: int) -> bytes:
    """
    Build a PDF grouping an axis title and a caption in one text block.

    Parameters
    ----------
    opening : str
        Figure or table caption opening.
    label_lines : int
        Number of short axis-title lines before the caption.

    Returns
    -------
    bytes
        PDF containing graphics, an axis title and a caption.
    """

    with pymupdf.open() as document:
        document.set_metadata({"subject": "doi:10.1234/caption-after-plot-label"})
        page = document.new_page()
        page.draw_rect(Rect(100, 80, 400, 180), color=(0, 0, 1), fill=(0.8, 0.8, 1))
        for i in range(label_lines):
            page.insert_text((190, 192 + 14 * i), "Time (min)", fontsize=10)
        page.insert_text(
            (60, 192 + 14 * label_lines),
            f"{opening} Measurements of the sample.",
            fontname="hebo",
            fontsize=10,
        )
        return document.tobytes()


@pytest.mark.parametrize("opening", ["Figure 1.", "Fig.S1.", "Table A2.1."])
@pytest.mark.parametrize("label_lines", [1, 2])
def test_caption_after_plot_labels(opening: str, label_lines: int) -> None:
    """Find captions without losing preceding labels from figure rectangles."""

    pdf = _plot_label_pdf(opening, label_lines)
    with pymupdf.open(stream=pdf) as document:
        raw = TextBlockPDF.from_dict(document[0].get_text("dict")["blocks"][0])
    assert len(raw.lines) == label_lines + 1
    assert raw.lines[0].text == "Time (min)"
    with ArticlePDF(pdf) as article:
        labels, caption = article._text_cache[0]
        assert article._raw_text_cache[0] == (labels, caption)
        for part in (labels, caption):
            assert ArticlePDF._normalize_caption_block(part) == (part,)
        assert labels.lines == raw.lines[:label_lines]
        assert caption.lines == raw.lines[label_lines:]
        assert labels.rect.y1 <= caption.rect.y0
        assert len(raw.lines) == label_lines + 1  # Original block is not mutated.
        if opening.startswith("Table"):
            found = article._table_captions_cache[0]
            assert [(c.label, c.text) for c in found] == [
                ("A2.1", "Measurements of the sample.")
            ]
        else:
            label = "S1" if opening.startswith("Fig.") else "1"
            assert article.figure_captions == {label: "Measurements of the sample."}
            assert article.figures[label].rect == Rect(100, 80, 400, 180) | labels.rect
            assert article.figures[label].caption.rect == caption.rect


@pytest.mark.parametrize(
    "case", ["prose", "wide", "same_row", "overlap", "caption_first", "earlier_prose"]
)
def test_unrelated_leading_text_is_not_split(case: str) -> None:
    """Do not promote an interior reference in prose or beside another line."""

    with pymupdf.open(stream=_plot_label_pdf("Figure 1.", 2)) as document:
        block = TextBlockPDF.from_dict(document[0].get_text("dict")["blocks"][0])
        first, second, caption = block.lines
        if case in ("prose", "earlier_prose"):
            first = replace(
                first,
                spans=[
                    replace(
                        first.spans[0],
                        text="A running paragraph discussing the measured results.",
                    )
                ],
            )
        elif case == "wide":
            first = replace(first, rect=Rect(60, first.rect.y0, 400, first.rect.y1))
        elif case == "same_row":
            first = replace(first, rect=caption.rect)
        elif case == "overlap":
            first = replace(first, rect=first.rect + (0, 25, 0, 25))
        lines = [first, caption] if case != "earlier_prose" else [first, second, caption]
        if case == "caption_first":
            lines = [caption, first]
        candidate = replace(block, lines=lines)
        assert ArticlePDF._normalize_caption_block(candidate) == (candidate,)
