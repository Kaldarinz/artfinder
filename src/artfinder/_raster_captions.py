"""Read positioned text from the lower strip of a figure image."""

import os
import subprocess
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pymupdf import Rect, csGRAY

if TYPE_CHECKING:
    # See ArticlePDF's workaround for PyMuPDF's Page type.
    class Page(Any): ...

else:
    from pymupdf import Page


@dataclass
class RasterCaption:
    text: str
    rect: Rect
    lines_no: int
    figure_rect: Rect


def _ocr_lines(tsv: str) -> list[tuple[Rect, str]]:
    """Combine Tesseract's word boxes into lines, in top-to-bottom order.

    Parameters
    ----------
    tsv : str
        Tesseract's positioned TSV output.

    Returns
    -------
    list[tuple[Rect, str]]
        Text lines and their enclosing rectangles in pixmap coordinates.
    """
    words: dict[tuple[str, ...], list[tuple[Rect, str]]] = defaultdict(list)
    for row in tsv.splitlines():
        fields = row.split("\t", 11)
        if len(fields) != 12 or fields[0] != "5" or not fields[-1].strip():
            continue
        x, y, width, height = map(int, fields[6:10])
        words[tuple(fields[1:5])].append(
            (Rect(x, y, x + width, y + height), fields[-1].strip())
        )
    lines: list[tuple[Rect, str]] = []
    for line_words in words.values():
        rect = Rect()
        for word_rect, _ in line_words:
            rect.include_rect(word_rect)
        lines.append((rect, " ".join(word for _, word in line_words)))
    return sorted(lines, key=lambda line: (line[0].y0, line[0].x0))


def read_raster_caption(
    page: Page, image: Rect, executable: str, opens_caption: Callable[[str], bool]
) -> RasterCaption | None:
    """Read a caption at the foot of an image, separated from the plot by whitespace.

    OCR only the lower third, so most plot text is never interpreted as a
    caption. The caption must be the last group of lines, not a figure
    reference in a plot title or annotation. Returned boxes are in unrotated
    page coordinates, like the rest of the extraction pipeline.

    Parameters
    ----------
    page : Page
        Page displaying the image.
    image : Rect
        Visible bounds of the image in unrotated page coordinates.
    executable : str
        Path to Tesseract.
    opens_caption : Callable[[str], bool]
        Test for a recognized caption opening.

    Returns
    -------
    RasterCaption | None
        The caption and its source figure's crop, if detected.
    """
    displayed = (image * page.rotation_matrix) & page.rect
    if displayed.is_empty:
        return None
    strip = Rect(displayed)
    strip.y0 += 2 * strip.height / 3
    dpi = 360
    scale = dpi / 72
    pix = page.get_pixmap(dpi=dpi, clip=strip, colorspace=csGRAY)
    if not pix.width or not pix.height:
        return None
    # A pale band across the entire image is evidence of a gap, even
    # when the OCR engine misses the axis label just above the caption.
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    blank_rows = (pixels >= 235).all(axis=1)
    edges = np.flatnonzero(np.diff(blank_rows, prepend=False, append=False)).tolist()
    gaps = [
        (top, bottom)
        for top, bottom in zip(edges[::2], edges[1::2])
        if bottom - top >= scale
    ]
    if not gaps or blank_rows.all():
        return None
    result = subprocess.run(
        [executable, "stdin", "stdout", "-l", "eng", "--psm", "11", "tsv"],
        input=pix.tobytes("png"), capture_output=True, check=True, timeout=10,
        # Callers may extract several PDFs in parallel.
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    lines = _ocr_lines(result.stdout.decode("utf-8"))
    for i, (first, text) in enumerate(lines):
        if not opens_caption(text):
            continue
        gap = next(
            (
                (top, min(bottom, first.y0))
                for top, bottom in reversed(gaps)
                if min(bottom, first.y0) - top >= scale
                and first.y0 - bottom <= 0.5 * first.height
            ),
            None,
        )
        if gap is None:
            continue
        bounds = Rect(first)
        parts = [text]
        last = first
        for rect, line_text in lines[i + 1:]:
            if (
                opens_caption(line_text)
                or rect.y0 < last.y1 - 0.25 * last.height
                or rect.y0 - last.y1 > max(last.height, rect.height)
                or min(first.x1, rect.x1) <= max(first.x0, rect.x0)
            ):
                break
            bounds.include_rect(rect)
            parts.append(line_text)
            last = rect
        if i + len(parts) != len(lines) or pix.height - bounds.y1 > 2 * first.height:
            continue

        bounds += (pix.x, pix.y, pix.x, pix.y)
        bounds /= scale
        bounds &= displayed
        figure = Rect(displayed)
        figure.y1 = (sum(gap) / 2 + pix.y) / scale
        if not bounds.is_empty and not figure.is_empty:
            return RasterCaption(
                " ".join(parts), bounds * page.derotation_matrix, len(parts),
                figure * page.derotation_matrix,
            )
    return None
