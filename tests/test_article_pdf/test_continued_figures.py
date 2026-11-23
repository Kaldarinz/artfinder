"""Explicit continuation captions link physical crops, not independent figures."""

from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest
from pymupdf import Pixmap, Rect

from artfinder.article_pdf import ArticlePDF
from tests.pdf_assertions import assert_stacked_png


PAPER = Path(__file__).parent / "article_pdfs" / "10.1038_s41598-024-55146-x.pdf"


def test_real_figure_stacks_both_placed_images(tmp_path: Path) -> None:
    """Check cold physical lookup and the exported pixels against the original PDF."""
    with ArticlePDF(PAPER) as article:
        panels: list[Pixmap] = []
        for n in (9, 8):  # Look up the continuation before visiting its owner.
            page = article.file[n]
            image, = page.get_image_info()
            rect = Rect(image["bbox"])
            assert article.get_figure_rects(n) == [rect]
            caption, = article.get_figure_caption_rects(n)
            assert caption.y0 > rect.y1
            panels.insert(0, page.get_pixmap(clip=rect, dpi=72))
        assert article.figure_count == 12
        assert article.get_figures(9) == {}
        figure = article.get_figures(8)["8"]
        assert figure.caption.text.startswith("EDX spectra and elemental mapping images")
        png = article.extract_figures("8", page_index=8, output_path=tmp_path, dpi=72)["8"]
    assert_stacked_png(png, panels, gap=6)


def _captioned_pages(captions: list[str]) -> bytes:
    """Build distinct page-local drawings, each with its own printed caption."""
    with pymupdf.open() as doc:
        for i, caption in enumerate(captions):
            page = doc.new_page()
            page.draw_rect(Rect(80, 100, 200 + 5 * i, 200), fill=(1, 0, 0))
            page.insert_text((80, 225), caption)
        return bytes(doc.tobytes())


@pytest.mark.parametrize("n", [1, 3], ids=["first-continuation", "last-continuation"])
def test_physical_crop_resolves_only_the_required_prefix(n: int) -> None:
    data = _captioned_pages(["Figure 1. Spectrum."] + ["Figure 1. (continued)"] * 3)
    with ArticlePDF(data, identifier="physical") as article:
        with patch.object(
            article._continued_parts, "factory", wraps=article._continued_parts_on_page
        ) as resolve:
            crop, = article.get_figure_rects(n, copy_rects=False)
            resolve.assert_called_once_with(n)  # no nested factory calls for earlier markers
        assert not article._figures_cache  # marking one crop did not assemble the whole figure
        assert set(article._figure_matches_cache) == set(range(n + 1))
        assert article.get_figures(0)["1"].following_parts[n - 1][1] is crop


def test_explicit_run_keeps_original_caption_and_anchor() -> None:
    data = _captioned_pages(["Fig. S2. Original legend.", "Fig. S2. Continued.",
                             "Fig. S2. (CONTINUED)"])
    with ArticlePDF(data, identifier="run") as article:
        assert article.figure_count == 1
        assert article.figure_captions == {"S2": "Original legend."}
        figure = article.figures["S2"]
        assert [n for n, _ in figure.following_parts] == [1, 2]
        assert len(article.get_figure_drawings("S2", page_index=0)) == 3
        replacement = Rect(90, 120, 165, 190)
        figure.following_parts = (figure.following_parts[0], (2, replacement))
        assert article.get_figure_rects(2, copy_rects=False)[0] is replacement
        figure.following_parts = ((2, replacement),)
        assert article.get_figure_rects(1) == []


@pytest.mark.parametrize("captions", [
    ["Figure 3. Original.", "Figure 4. (continued)"],
    ["Figure 3. Original.", "Figure 4. Other.", "Figure 3. (continued)"],
    ["Figure 3. Original.", "Figure 3. Continued growth of the sample."],
    ["Figure 3. (continued)", "Figure 3. (continued)"],
])
def test_ambiguous_or_orphan_markers_remain_local(captions: list[str]) -> None:
    with ArticlePDF(_captioned_pages(captions), identifier="separate") as article:
        assert article.figure_count == len(captions)
        for n in range(len(captions)):
            figure, = article.get_figures(n).values()
            assert not figure.following_parts
