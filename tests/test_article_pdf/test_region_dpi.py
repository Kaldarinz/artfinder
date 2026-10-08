"""
Tests for the resolution an arbitrary page region is rasterized at.
"""

from typing import Any

import pymupdf
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF

# Pages are typed `Any`: mypy cannot see PyMuPDF's `Page` (see
# docs/test_setup.md).

# A 150 x 150 pixel raster placed over 72 x 72 points reads at 150 dpi.
RASTER_RECT = Rect(100, 100, 172, 172)


def _raster(page: Any, rect: Rect = RASTER_RECT) -> None:
    """Place a grey 150-dpi raster image at `rect` on `page`."""

    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 150, 150), False)
    pixmap.clear_with(128)
    page.insert_image(rect, pixmap=pixmap)


def _document() -> tuple[pymupdf.Document, Any]:
    """A new one-page A4 document and its page."""

    doc = pymupdf.open()
    return doc, doc.new_page(width=595, height=842)


def test_a_raster_alone_takes_its_own_resolution() -> None:
    doc, page = _document()
    _raster(page)
    with ArticlePDF(doc.tobytes()) as pdf:
        assert pdf.region_dpi(0, Rect(90, 90, 200, 200)) == 150


def test_a_visible_drawing_beside_a_raster_takes_the_vector_rate() -> None:
    doc, page = _document()
    _raster(page)
    page.draw_rect(Rect(180, 100, 250, 170), color=(1, 0, 0), fill=(0, 0, 1))
    with ArticlePDF(doc.tobytes()) as pdf:
        assert pdf.region_dpi(0, Rect(90, 90, 260, 200)) == 300


def test_hairline_axes_alone_take_the_vector_rate() -> None:
    # Straight lines have rectangles of zero width or height, and the
    # x axis runs on past the region: neither lies inside it.
    doc, page = _document()
    _raster(page)
    page.draw_line((95, 180), (400, 180), color=(0, 0, 0), width=0.5)
    page.draw_line((95, 95), (95, 180), color=(0, 0, 0), width=0.5)
    page.draw_line((120, 180), (120, 183), color=(0, 0, 0), width=0.5)
    with ArticlePDF(doc.tobytes()) as pdf:
        assert pdf.region_dpi(0, Rect(90, 90, 200, 200)) == 300


def test_a_white_background_is_not_a_drawing() -> None:
    doc, page = _document()
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
    _raster(page)
    with ArticlePDF(doc.tobytes()) as pdf:
        assert pdf.region_dpi(0, Rect(90, 90, 200, 200)) == 150


def test_text_alone_takes_the_vector_rate() -> None:
    doc, page = _document()
    page.insert_text((100, 400), "Concentration 0.1 mg/mL", fontsize=10)
    _raster(page)
    with ArticlePDF(doc.tobytes()) as pdf:
        assert pdf.region_dpi(0, Rect(90, 380, 300, 420)) == 300


def test_a_region_on_a_rotated_page_is_given_as_displayed() -> None:
    doc, page = _document()
    _raster(page)
    vector = Rect(100, 500, 172, 572)
    page.draw_rect(vector, color=(1, 0, 0), fill=(0, 0, 1))
    page.set_rotation(90)
    raster_shown = RASTER_RECT * page.rotation_matrix
    vector_shown = vector * page.rotation_matrix
    with ArticlePDF(doc.tobytes()) as pdf:
        # Taken as unrotated coordinates, the raster's displayed box
        # would hold neither the raster nor the drawing.
        assert pdf.region_dpi(0, raster_shown) == 150
        assert pdf.region_dpi(0, vector_shown) == 300
