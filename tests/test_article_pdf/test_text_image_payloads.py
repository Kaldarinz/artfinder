"""Images may separate text blocks even though their pixels are not text."""

import pymupdf
import pytest
from pymupdf import Rect, TextPage

from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import TextBlockPDF


@pytest.mark.parametrize(
    "line_gap", [12, 40], ids=["image-separates-lines", "independent-blocks"]
)
def test_text_layout_with_intervening_image(
    line_gap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep native text geometry and omit image payloads when layout permits."""
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Before")
        page.insert_image(
            Rect(300, 100, 350, 150), stream=b"P6\n4 4\n255\n" + b"\x00\xff\x00" * 16
        )
        page.insert_text((72, 72 + line_gap), "After")
        expected = tuple(
            TextBlockPDF.from_dict(b)
            for b in page.get_text("dict")["blocks"]
            if b["type"] == 0
        )
        data = doc.tobytes()

    extract = TextPage.extractDICT
    image_bytes = 0

    def counted_extract(page: TextPage, cb: Rect | None = None, sort: bool = False) -> dict:
        nonlocal image_bytes
        result = extract(page, cb=cb, sort=sort)
        image_bytes += sum(len(b["image"]) for b in result["blocks"] if b["type"] == 1)
        return result

    monkeypatch.setattr(TextPage, "extractDICT", counted_extract)
    with ArticlePDF(data, identifier="interleaved") as article:
        assert len(expected) == 2
        assert article._raw_text_cache[0] == expected
    if line_gap == 40:
        assert image_bytes == 0
