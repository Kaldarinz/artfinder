"""Regression tests for panel letters extracted before a side caption."""

from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import TextBlockPDF


def _side_caption_pdf(caption_x: float = 50) -> bytes:
    """
    Build a PDF with panel letters and a side caption in one text block.

    Parameters
    ----------
    caption_x : float, default=50
        Left edge of the caption, beside the figure on either side.

    Returns
    -------
    bytes
        PDF whose extraction order puts panel letters before its caption.
    """

    with pymupdf.open() as document:
        document.set_metadata({"subject": "doi:10.1234/side-caption"})
        page = document.new_page()
        page.draw_rect(Rect(200, 200, 400, 320), color=(0, 0, 1))
        page.insert_text((205, 205), "(b)", fontsize=10)
        page.insert_text((305, 210), "(a)", fontsize=10)
        page.insert_text((caption_x, 205), "Fig. 3 Measurements", fontsize=10)
        page.insert_text((caption_x, 217), "of the sample.", fontsize=10)
        return document.tobytes()


@pytest.mark.parametrize("caption_x", [50, 450])
def test_panel_letters_do_not_hide_side_caption(caption_x: float) -> None:
    """Recover the caption and retain its panel letters as figure content."""

    pdf = _side_caption_pdf(caption_x)
    with pymupdf.open(stream=pdf) as document:
        raw = document[0].get_text("dict")["blocks"]
        assert len(raw) == 1
        assert [line["spans"][0]["text"] for line in raw[0]["lines"]] == [
            "(b)",
            "(a)",
            "Fig. 3 Measurements",
            "of the sample.",
        ]
    with ArticlePDF(pdf) as article:
        blocks = article._text_cache[0]
        assert [block.text for block in blocks] == [
            "(b)",
            "(a)",
            "Fig. 3 Measurements of the sample.",
        ]
        assert article.figure_captions == {"3": "Measurements of the sample."}
        figure = article.figures["3"]
        assert figure.rect.contains(blocks[0].rect)
        assert figure.rect.contains(blocks[1].rect)
        assert not figure.rect.intersects(figure.caption.rect)


@pytest.mark.parametrize(
    "case", ["prose", "number", "overlap", "above", "far_above", "caption_first"]
)
def test_unrelated_leading_lines_are_not_split(case: str) -> None:
    """Only detach parenthesized panel letters outside the caption's span."""

    with pymupdf.open(stream=_side_caption_pdf()) as document:
        block = TextBlockPDF.from_dict(document[0].get_text("dict")["blocks"][0])
    first, second, opening, tail = block.lines
    if case in {"prose", "number"}:
        first = replace(
            first,
            spans=[replace(first.spans[0], text="See also" if case == "prose" else "(1)")],
        )
    elif case == "overlap":
        first = replace(first, rect=Rect(opening.rect))
    elif case == "above":
        first = replace(first, rect=opening.rect + (0, -15, 0, -15))
    elif case == "far_above":
        first = replace(first, rect=first.rect + (0, -30, 0, -30))
    lines = [opening, tail] if case == "caption_first" else [first, second, opening, tail]
    candidate = replace(block, lines=lines)
    assert ArticlePDF._split_caption_panel_labels(candidate) == (candidate,)


def test_nanofabrication_figure_three() -> None:
    """Recover both panels of the real article without including its caption."""

    path = Path(__file__).parent / "article_pdfs" / "nanofabrication with pulsed lasers.pdf"
    with ArticlePDF(path) as article:
        figure = article.figures["3"]
        assert article.figure_captions["3"] == (
            "a Typical image of Si-based nanostructures prepared by laser "
            "plasma-assisted treatment of a Si wafer; b Photoluminescence spectra "
            "from laser plasma-treated nanostructured Si spots for different "
            "pumping wavelengths (325 and 488 nm)"
        )
        assert figure.rect.contains(Rect(204, 581, 361, 699))
        assert figure.rect.contains(Rect(378.4, 578, 543, 713))
        assert figure.rect.x0 > figure.caption.rect.x1
        assert len(article.figures) == 7
