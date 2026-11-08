"""Regression tests for internal caption cuts and neighbouring captions."""

from pathlib import Path

import pymupdf
import pytest

from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import Rect, TextBlockPDF, TextLinePDF


def _line(text: str, rect: tuple[float, float, float, float]) -> TextLinePDF:
    """Build a line with one ordinary caption-font span.

    Parameters
    ----------
    text : str
        Line text.
    rect : tuple[float, float, float, float]
        Line and span bounding box.

    Returns
    -------
    TextLinePDF
        Synthetic caption line.
    """
    return TextLinePDF.from_dict(
        {
            "bbox": rect,
            "spans": [{"bbox": rect, "text": text, "size": 9, "font": "Caption"}],
        }
    )


def _block(lines: list[TextLinePDF]) -> TextBlockPDF:
    """Build a block enclosing its lines.

    Parameters
    ----------
    lines : list[TextLinePDF]
        Lines to include.

    Returns
    -------
    TextBlockPDF
        Synthetic text block.
    """
    rect = Rect()
    for line in lines:
        rect.include_rect(line.rect)
    return TextBlockPDF(rect=rect, lines=lines)


def test_caption_spacing_tolerates_small_font_metric_differences() -> None:
    """Keep regularly spaced text whose span boxes differ slightly in height."""
    # Core rectangles from the reported Figure 8: the first line's font
    # metrics shorten the measured first pitch from about 12.8 to 11.1 pt.
    block = _block(
        [
            _line("Figure 8. First line", (100, 425.45, 500, 436.92)),
            _line("second line", (100, 438.95, 500, 448.00)),
            _line("third line", (100, 451.09, 500, 460.82)),
            _line("last line.", (100, 463.91, 200, 473.62)),
        ]
    )

    kept, cut = ArticlePDF._cut_at_blank_line(block)

    assert not cut
    assert kept == block


@pytest.mark.parametrize("kind", ["Figure", "Table"])
@pytest.mark.parametrize("cut_at", ["blank_line", "table_row"])
def test_internal_cut_does_not_skip_next_caption(
    tmp_path: Path, kind: str, cut_at: str
) -> None:
    """A cut ends this caption, not processing of the next text block."""
    lines = [
        _line(f"{kind} 1. First caption", (100, 100, 400, 110)),
        _line("continued.", (100, 113, 400, 123)),
    ]
    if cut_at == "blank_line":
        lines.append(_line("A following paragraph.", (100, 145, 400, 155)))
    else:
        lines.extend(
            [
                _line("Cell A", (100, 126, 150, 136)),
                _line("Cell B", (300, 126, 350, 136)),
            ]
        )
    blocks = [
        _block(lines),
        _block([_line(f"{kind} 2. Second caption.", (100, 300, 400, 310))]),
    ]
    source = tmp_path / "captions.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(source)

    with ArticlePDF(source) as article:
        article._text_cache[0] = blocks
        cache = (
            article._figure_captions_cache
            if kind == "Figure"
            else article._table_captions_cache
        )
        captions = {caption.label: caption.text for caption in cache[0]}

    assert captions == {"1": "First caption continued.", "2": "Second caption."}
