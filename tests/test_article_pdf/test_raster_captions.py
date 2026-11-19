"""Regression tests for figure captions printed as pixels instead of text."""

from pathlib import Path

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF


def _raster_pdf(path: Path, *, native: bool = False, annotation: bool = False) -> None:
    """Write a PDF with an image carrying either a caption or a plot annotation."""
    with pymupdf.open() as source, pymupdf.open() as doc:
        plot = source.new_page(width=280, height=170)
        plot.draw_rect(Rect(25, 20, 255, 125), color=(0, 0, 0))
        plot.draw_line((30, 110), (240, 30))
        plot.insert_text((10, 155 if not annotation else 123), "Fig. 3. Embedded plot.")
        if annotation:
            plot.insert_text((90, 148), "Time in seconds")
        page = doc.new_page(width=400, height=400)
        page.insert_image(Rect(60, 60, 340, 230), pixmap=plot.get_pixmap(dpi=144))
        if native:
            page = doc.new_page(width=400, height=400)
            page.draw_rect(Rect(60, 60, 340, 200), color=(0, 0, 0))
            page.insert_text((60, 230), "Fig. 3. Native caption.")
        doc.save(path)


def test_preserves_clear_native_caption(tmp_path: Path, tesseract: None) -> None:
    """A duplicate OCR label does not replace a clear native caption."""
    pdf = tmp_path / "native.pdf"
    _raster_pdf(pdf, native=True)
    with ArticlePDF(pdf) as article:
        assert article.figure_captions == {"3": "Native caption."}
        assert not article.get_figure_caption_rects(0)
        assert article.get_figure_caption_rects(1)


def test_extracts_components_of_raster_figure(tmp_path: Path, tesseract: None) -> None:
    """The crop excludes caption pixels but still has the source image component."""
    pdf = tmp_path / "raster.pdf"
    _raster_pdf(pdf)
    with ArticlePDF(pdf) as article:
        assert article.figure_captions == {"3": "Embedded plot."}
        figure = article.figures["3"]
        assert 185 < figure.rect.y1 < figure.caption.rect.y0 < 215
        assert len(article.get_figure_images("3")) == 1
        result = article.extract_figures("3", output_path=tmp_path)
        assert result["3"].is_file()


def test_ignores_raster_plot_annotation(tmp_path: Path, tesseract: None) -> None:
    """A label-like annotation inside the plot is not its caption."""
    pdf = tmp_path / "annotation.pdf"
    _raster_pdf(pdf, annotation=True)
    with ArticlePDF(pdf) as article:
        assert not article.figure_captions


def test_no_tesseract_still_extracts_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native figure extraction does not require an OCR engine."""
    pdf = tmp_path / "no-ocr.pdf"
    _raster_pdf(pdf, native=True)
    monkeypatch.setattr("artfinder.article_pdf.which", lambda _: None)
    with ArticlePDF(pdf) as article:
        assert article.figure_captions == {"3": "Native caption."}
