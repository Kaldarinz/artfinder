"""Regression tests for balanced captions continued in the next column."""

from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import TextBlockPDF


def _column(texts: list[str], x: float, y: float = 400) -> TextBlockPDF:
    """
    Build a justified caption column with regular line spacing.

    Parameters
    ----------
    texts : list[str]
        Lines in reading order.
    x, y : float
        Left and top coordinates of the column.

    Returns
    -------
    TextBlockPDF
        Column with 240-point-wide lines at a 12-point pitch.
    """

    return TextBlockPDF.from_dict(
        {
            "type": 0,
            "lines": [
                {
                    "bbox": (x, y + 12 * i, x + 240, y + 12 * i + 10),
                    "spans": [
                        {
                            "bbox": (x, y + 12 * i, x + 240, y + 12 * i + 10),
                            "text": text,
                            "font": "Caption",
                            "size": 9,
                        }
                    ],
                }
                for i, text in enumerate(texts)
            ],
        }
    )


@pytest.mark.parametrize("kind", ["Figure", "Table"])
@pytest.mark.parametrize("hyphenated", [False, True])
def test_caption_continues_in_column_order(kind: str, hyphenated: bool) -> None:
    """Read down the left column, then the right, preserving caption bounds."""

    left = _column(
        [
            f"{kind} 6. Measurements of",
            "the sample showing",
            "the dependence on pres-" if hyphenated else "the dependence on",
        ],
        50,
    )
    right = _column(
        [
            "sure during deposition;" if hyphenated else "pressure during deposition;",
            "inset shows the structure",
            "and the final peak position.",
        ],
        310,
    )
    next_caption = _column([f"{kind} 7. Separate caption."], 50, 500)
    with pymupdf.open() as document:
        document.set_metadata({"subject": "doi:10.1234/caption-columns"})
        document.new_page()
        pdf = document.tobytes()
    with ArticlePDF(pdf) as article:
        article._text_cache[0] = (left, right, next_caption)
        captions = (
            article._figure_captions_cache[0]
            if kind == "Figure"
            else article._table_captions_cache[0]
        )
        assert {caption.label: caption.text for caption in captions} == {
            "6": "Measurements of the sample showing the dependence on pressure "
            "during deposition; inset shows the structure and the final peak position.",
            "7": "Separate caption.",
        }
        assert captions[0].rect == Rect(50, 400, 550, 434)
        assert captions[0].lines_no == 6


@pytest.mark.parametrize(
    "case",
    [
        "complete_sentence",
        "uppercase",
        "figure_caption",
        "table_caption",
        "different_font",
        "different_size",
        "different_width",
        "short_last_line",
        "far_column",
        "overlapping_column",
        "different_rows",
        "different_line_count",
        "single_line",
        "empty",
    ],
)
def test_neighbouring_text_is_not_a_column_continuation(case: str) -> None:
    """Require unfinished text, matching typography and balanced column geometry."""

    left_text = ["Figure 6. Measurements of", "the sample showing", "the dependence on"]
    right_text = [
        "pressure during deposition;",
        "inset shows the structure",
        "and the final peak position.",
    ]
    if case == "complete_sentence":
        left_text[-1] += "."
    elif case == "uppercase":
        right_text[0] = "An independent paragraph."
    elif case == "figure_caption":
        right_text[0] = "fig. 7. Independent caption"
    elif case == "table_caption":
        right_text[0] = "table 7. Independent caption"
    elif case == "different_line_count":
        right_text.pop()
    elif case == "single_line":
        left_text, right_text = left_text[:1], right_text[:1]
    elif case == "empty":
        right_text = []
    left = _column(left_text, 50)
    right = _column(right_text, 310)
    if case in {"different_font", "different_size"}:
        right = replace(
            right,
            lines=[
                replace(
                    line,
                    spans=[
                        replace(
                            span,
                            font="Body" if case == "different_font" else span.font,
                            size=12 if case == "different_size" else span.size,
                        )
                        for span in line.spans
                    ],
                )
                for line in right.lines
            ],
        )
    elif case == "different_width":
        right = replace(right, rect=right.rect + (0, 0, -50, 0))
    elif case == "short_last_line":
        left = replace(
            left,
            lines=left.lines[:-1]
            + [replace(left.lines[-1], rect=left.lines[-1].rect + (0, 0, -60, 0))],
        )
    elif case == "far_column":
        right = _column(right_text, 410)
    elif case == "overlapping_column":
        right = _column(right_text, 250)
    elif case == "different_rows":
        right = _column(right_text, 310, 403)
    assert not ArticlePDF._continues_caption_column(left, right)


def test_nanofabrication_figure_six_caption_bounds() -> None:
    """The real caption covers both columns without changing the figure bounds."""

    path = Path(__file__).parent / "article_pdfs" / "nanofabrication with pulsed lasers.pdf"
    with ArticlePDF(path) as article:
        figure = article.figures["6"]
        assert figure.caption.rect.contains(Rect(51.1, 677.4, 544.2, 715.7))
        assert figure.caption.lines_no == 8
        assert figure.caption.text.endswith(
            "d Dependence of the position of PL peak from laser-ablated ﬁlms "
            "on the gas pressure during the deposition"
        )
        assert figure.rect.y1 < figure.caption.rect.y0
