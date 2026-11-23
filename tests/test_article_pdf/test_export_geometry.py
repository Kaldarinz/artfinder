"""PNG layers use the selected page's coordinate system and image resources."""

from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest
from pymupdf import Pixmap, Rect

from artfinder.article_pdf import ArticlePDF


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_all_layers_use_figure_page_geometry(tmp_path: Path, rotation: int) -> None:
    """Later, larger pages keep their dimensions and rotation in every PNG layer."""
    bounds = Rect(350, 300, 510, 380)
    with pymupdf.open() as doc:
        doc.new_page(width=250, height=250)
        page = doc.new_page(width=600, height=750)
        page.insert_image(Rect(350, 300, 430, 380), stream=b"P6\n2 2\n255\n" + b"\xff\x00\x00" * 4)
        page.draw_rect(Rect(430, 300, 510, 380), fill=(0, 1, 0))
        page.insert_text((440, 330), "A")
        page.insert_text((350, 400), "Figure 1. Example.")
        page.set_rotation(rotation)
        expected = page.get_pixmap(dpi=72, clip=bounds * page.rotation_matrix)
        data = doc.tobytes()
    with ArticlePDF(data, identifier="mixed") as article:
        assert article.figures["1"].rect == bounds
        paths = article.extract_figures("1", output_path=tmp_path, dpi=72, extract_bases=True)
    full = Pixmap(str(paths["1"]))
    assert (full.width, full.height) == (expected.width, expected.height)
    assert full.samples == expected.samples
    for suffix in ("image", "vectors", "text"):
        layer = Pixmap(str(tmp_path / f"mixed_fig_1_{suffix}.png"))
        assert (layer.width, layer.height) == (expected.width, expected.height)
        if suffix == "text":
            assert min(layer.samples) < 128
        else:
            color = (255, 0, 0) if suffix == "image" else (0, 255, 0)
            for x in (layer.width // 4, 3 * layer.width // 4):
                for y in (layer.height // 4, 3 * layer.height // 4):
                    assert (layer.pixel(x, y) == color) == (full.pixel(x, y) == color)


def test_srep32631_exports_match_source_images(tmp_path: Path) -> None:
    """Each exported SI figure matches the entire placed image on its source page."""
    pdf = Path(__file__).parent / "article_pdfs" / "10.1038_srep32631_si.pdf"
    with ArticlePDF(pdf) as article:
        paths = article.extract_figures(output_path=tmp_path, dpi=72)
        for page_no, labels in enumerate((("S1", "S2"), ("S3", "S4", "S5"), ("S6", "S7")), start=1):
            page = article.file[page_no]
            for label, image in zip(labels, page.get_image_info(), strict=True):
                source = page.get_pixmap(dpi=72, clip=Rect(image["bbox"]))
                png = Pixmap(str(paths[label]))
                assert (png.width, png.height, png.digest) == (source.width, source.height, source.digest), label


def test_empty_layers_allocate_no_scratch_pages(tmp_path: Path) -> None:
    """A vectors-only figure needs one scratch document, not one per layer."""
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.draw_rect(Rect(100, 150, 260, 240), fill=(0, 1, 0))
        page.insert_text((100, 260), "Figure 1. Example.")
        data = doc.tobytes()
    with ArticlePDF(data, identifier="vectors") as article:
        assert set(article.figures) == {"1"}
        with patch.object(pymupdf, "open", wraps=pymupdf.open) as opened:
            article.extract_figures(output_path=tmp_path, dpi=72, extract_bases=True)
            opened.assert_called_once_with()
    assert {p.name for p in tmp_path.iterdir()} == {"vectors_fig_1.png", "vectors_fig_1_vectors.png"}


def test_inline_payload_lookup_is_per_page_per_export(tmp_path: Path) -> None:
    """Figures share their page's payload lookup but never use another page's bytes."""
    colors = [(255, 0, 0), (0, 0, 255)]
    with pymupdf.open() as doc:
        for i, color in enumerate(colors):
            page = doc.new_page(width=600, height=800)
            stream = doc.get_new_xref()
            doc.update_object(stream, "<<>>")
            doc.update_stream(stream, b"\n".join(
                f"q 100 0 0 90 80 {800 - y - 90} cm BI /W 1 /H 1 /BPC 8 /CS /RGB ID ".encode()
                + bytes(color) + b" EI Q" for y in (120, 340)
            ))
            page.set_contents(stream)
            for j, y in enumerate((120, 340), start=1):
                page.insert_text((80, y + 108), f"Figure {2 * i + j}. Example.")
        data = doc.tobytes()
    with ArticlePDF(data, identifier="inline") as article:
        assert list(article.figures) == ["1", "2", "3", "4"]
        with patch.object(pymupdf.TextPage, "extractDICT", autospec=True,
                          side_effect=pymupdf.TextPage.extractDICT) as extract:
            for count in (2, 4):
                paths = article.extract_figure_images(output_path=tmp_path, dpi=72)
                assert extract.call_count == count
                assert [Pixmap(str(p)).pixel(50, 45) for p in paths] == [c for c in colors for _ in range(2)]


def test_inline_image_layer(tmp_path: Path) -> None:
    """Two inline images without xrefs retain their distinct pixel payloads."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=750)
        stream = doc.get_new_xref()
        doc.update_object(stream, "<<>>")
        doc.update_stream(stream, b"q 80 0 0 80 80 520 cm BI /W 1 /H 1 /BPC 8 /CS /RGB ID "
                          b"\xff\x00\x00 EI Q\nq 80 0 0 80 160 520 cm BI /W 1 /H 1 /BPC 8 /CS /RGB ID "
                          b"\x00\x00\xff EI Q")
        page.set_contents(stream)
        page.insert_text((80, 250), "Figure 1. Two panels.")
        data = doc.tobytes()
    with ArticlePDF(data, identifier="inline") as article:
        assert [im.xref for im in article.get_figure_images("1")] == [0, 0]
        path, = article.extract_figure_images("1", output_path=tmp_path, dpi=72)
    pix = Pixmap(str(path))
    assert pix.pixel(40, 40) == (255, 0, 0)
    assert pix.pixel(120, 40) == (0, 0, 255)
