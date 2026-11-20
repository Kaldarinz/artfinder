"""Page-scoped PNG exports when the same caption label appears twice."""

from pathlib import Path

import pymupdf
import pytest
from pymupdf import Pixmap, Rect

from artfinder.article_pdf import ArticlePDF


def test_page_scoped_export_with_duplicate_label(tmp_path: Path) -> None:
    """Render the chosen page's figure and components at its own raster DPI."""
    with pymupdf.open() as doc:
        for y, samples, rgb in [(100, 216, (255, 0, 0)), (350, 432, (0, 0, 255))]:
            page = doc.new_page()
            image = f"P6\n{samples} {samples}\n255\n".encode() + bytes(rgb) * samples**2
            page.insert_image(Rect(100, y, 172, y + 72), stream=image)
            page.draw_rect(Rect(172, y, 244, y + 72), fill=(0, 1, 0))
            page.insert_text((205, y + 35), "A")
            page.insert_text((100, y + 89), "Figure 1. Example.")
        pdf = doc.tobytes()

    with ArticlePDF(pdf, identifier="dup") as article:
        paths = article.extract_figures(
            "1", page_index=0, output_path=tmp_path, extract_bases=True
        )
        # An unscoped request still warns and uses the last occurrence.
        with pytest.warns(UserWarning, match="Duplicate figure label 1"):
            last = article.extract_figures("1", output_path=tmp_path / "all", dpi=72)
        assert Pixmap(str(last["1"])).pixel(36, 36) == (0, 0, 255)

    # Each panel is 72 pt wide. The first page's 216-pixel raster sets 216 DPI.
    full = Pixmap(str(paths["1"]))
    assert full.width == 432
    assert full.pixel(108, 108) == (255, 0, 0)
    for suffix, x, color in [("image", 108, (255, 0, 0)), ("vectors", 324, (0, 255, 0))]:
        layer = Pixmap(str(tmp_path / f"dup_fig_1_{suffix}.png"))
        assert layer.pixel(x, 108) == color
    text = Pixmap(str(tmp_path / "dup_fig_1_text.png"))
    assert min(text.samples) < 128  # The panel letter, not an empty white crop.
