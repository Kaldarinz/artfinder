"""Regression tests for white title text overprinted on caption labels."""

from dataclasses import replace

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import TextBlockPDF


def _caption_pdf(opening: str, title_lines: int = 1) -> bytes:
    """
    Build a PDF whose caption starts behind white overprinted title lines.

    Parameters
    ----------
    opening : str
        Figure or table caption label.
    title_lines : int, default=1
        Number of overprinted title lines, all on the label's row.

    Returns
    -------
    bytes
        PDF containing a graphic and its caption.
    """

    with pymupdf.open() as document:
        document.set_metadata({"subject": "doi:10.1234/overlaid-caption"})
        page = document.new_page()
        page.draw_rect(Rect(100, 200, 400, 380), color=(0, 0, 1), fill=(0.8, 0.8, 1))
        for _ in range(title_lines):
            page.insert_text(
                (100, 405), "Overprinted caption title", fontsize=10, color=(1, 1, 1)
            )
        page.insert_text((100, 405), opening, fontname="hebo", fontsize=10)
        page.insert_text((100, 417), "Measurements of the sample.", fontsize=10)
        return document.tobytes()


@pytest.mark.parametrize("title_lines", [1, 2])
@pytest.mark.parametrize(
    "opening,label", [("Figure 1", "1"), ("Fig.S1", "S1"), ("Table A2.1", "A2.1")]
)
def test_overprinted_title_does_not_hide_caption(
    opening: str, label: str, title_lines: int
) -> None:
    """Find figure and table captions without including the white title."""

    with ArticlePDF(_caption_pdf(opening, title_lines)) as article:
        block = article._text_cache[0][0]
        assert len(block.lines) == title_lines + 2
        assert block.lines[0].text == "Overprinted caption title"
        if opening.startswith("Table"):
            assert {
                caption.label: caption.text
                for caption in article._table_captions_cache[0]
            } == {label: "Measurements of the sample."}
        else:
            assert article.figure_captions == {label: "Measurements of the sample."}
            assert set(article.figures) == {label}
            assert article.figures[label].rect == Rect(100, 200, 400, 380)


@pytest.mark.parametrize(
    "case", ["nonwhite", "mixed", "above", "beside", "caption_first"]
)
def test_unrelated_text_is_not_stripped(case: str) -> None:
    """Require white-only leading text at the actual label's position."""

    with ArticlePDF(_caption_pdf("Figure 1")) as article:
        block = article._text_cache[0][0]
        title, label, caption = block.lines
        if case == "nonwhite":
            title = replace(title, spans=[replace(title.spans[0], color=(0, 0, 0))])
        elif case == "mixed":
            title = replace(
                title, spans=title.spans + [replace(title.spans[0], color=(0, 0, 0))]
            )
        elif case == "above":
            title = replace(title, rect=title.rect + (0, -12, 0, -12))
        elif case == "beside":
            title = replace(title, rect=title.rect + (200, 0, 200, 0))
        lines = [label, caption] if case == "caption_first" else [title, label, caption]
        candidate = TextBlockPDF(rect=block.rect, lines=lines)
        assert ArticlePDF._strip_overlaid_caption_title(
            candidate, ArticlePDF.CAPTION_PATTERN
        ) is candidate


def test_white_caption_on_dark_background_is_preserved() -> None:
    """Do not generally discard white text, which can be visible on dark fills."""

    with pymupdf.open() as document:
        document.set_metadata({"subject": "doi:10.1234/white-caption"})
        page = document.new_page()
        page.draw_rect(Rect(90, 380, 400, 430), fill=(0, 0, 0))
        page.insert_text(
            (100, 405), "Figure 1 Measurements.", fontsize=10, color=(1, 1, 1)
        )
        with ArticlePDF(document.tobytes()) as article:
            assert article.figure_captions == {"1": "Measurements."}
