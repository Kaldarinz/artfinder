"""Physical marks and single-PNG exports of panel sequences split over a page break."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from weakref import ReferenceType, ref

import pymupdf
import pytest
from pymupdf import Pixmap, Rect

from artfinder.article_pdf import ArticlePDF, Page


SI = Path(__file__).parent / "article_pdfs" / "10.1021_acsomega.5c05967_si.pdf"
# Zero-based physical pages. Measured from the labelled plots on the original
# pages; S3's left and right columns must read row-wise, not (a,c,b,d).
PRECEDING: dict[str, tuple[int, Rect]] = {
    label: (page, Rect(bounds))
    for label, parts in json.loads(SI.with_name("preceding_figure_parts.json").read_text())[SI.name].items()
    for page, bounds in parts
}
PRIMARY = {label: Rect(bounds)
           for label, bounds in json.loads(SI.with_name("figure_rects.json").read_text())[SI.name].items()}


def _coords(rect: Rect) -> tuple[float, float, float, float]:
    return rect.x0, rect.y0, rect.x1, rect.y1


def test_split_figure_parts_and_clip() -> None:
    """Look up a preceding part cold; the caption and primary crop do not move."""
    with ArticlePDF(SI) as article:
        crop, = article.get_figure_rects(3)
        assert _coords(crop) == pytest.approx(_coords(PRECEDING["S3"][1]), abs=0.01)
        crop.x1 = 0  # default is a copy, not the cached part
        assert set(article.figures) == {f"S{i}" for i in range(1, 6)}
        for label, (n, expected) in PRECEDING.items():
            figure = article.get_figures(n + 1)[label]
            (page_no, part), = figure.preceding_parts
            assert page_no == n
            assert _coords(part) == pytest.approx(_coords(expected), abs=0.01)
            assert article.get_figure_rects(n, clip=part) == [part]
            assert _coords(figure.rect) == pytest.approx(_coords(PRIMARY[label]), abs=0.01)
        assert not article.figures["S5"].preceding_parts


@pytest.mark.parametrize("page_number", [None, 4], ids=["whole-document", "page-five-only"])
def test_marks_are_drawn_on_each_physical_page(tmp_path: Path, page_number: int | None) -> None:
    """Page five gets both S3(e,f) and S4(a), in two disjoint boxes."""
    with ArticlePDF(SI) as article:
        output = article.mark("figures", page_number=page_number, output_path=tmp_path,
                              stroke_color=(1, 0, 1), stroke_width=1.75)
    expected = {3: [PRECEDING["S3"][1]], 4: [PRIMARY["S3"], PRECEDING["S4"][1]],
                5: [PRIMARY["S4"], PRIMARY["S5"]]}
    with pymupdf.open(output) as marked:
        for n, rects in expected.items():
            actual = [d["rect"] for d in marked[n].get_drawings()
                      if d["color"] == (1, 0, 1) and d["width"] == 1.75]
            if page_number is not None and n != page_number:
                assert not actual
                continue
            assert len(actual) == len(rects)
            for got, wanted in zip(actual, rects):
                assert _coords(got) == pytest.approx(_coords(wanted), abs=0.01)


@pytest.mark.parametrize("label", ["S3", "S4"])
def test_png_stacks_the_original_page_pixels(tmp_path: Path, label: str) -> None:
    """Check the entire composed image, including white padding, against both source pages."""
    n, _ = PRECEDING[label]
    with ArticlePDF(SI) as article:
        f = article.get_figures(n + 1)[label]
        png = article.extract_figures(label, page_index=n + 1, output_path=tmp_path, dpi=96)[label]
        parts = [article.file[i].get_pixmap(dpi=96, clip=r)
                 for i, r in (*f.preceding_parts, (n + 1, f.rect))]
    assert list(tmp_path.iterdir()) == [png]
    out = Pixmap(str(png))
    assert out.width == max(p.width for p in parts)
    assert out.height == sum(p.height for p in parts) + 8  # 6 pt at 96 DPI
    pixels = bytearray(b"\xff" * (out.stride * out.height))
    top = 0
    for p in parts:
        left = (out.width - p.width) // 2
        data = p.samples
        for row in range(p.height):
            start = (top + row) * out.stride + 3 * left
            pixels[start:start + p.stride] = data[row * p.stride:(row + 1) * p.stride]
        top += p.height + 8
    assert out.samples == pixels


def _panel_pdf(case: str, *, rotate_first: bool = False) -> bytes:
    """Build a vector panel followed by a low-DPI raster panel and its caption."""
    with pymupdf.open() as doc:
        for i in range(2):
            page = doc.new_page(width=400, height=600)
            y = 90 if i else 220
            x = 150 if i and case == "misaligned" else 80
            rect = Rect(x, y, x + 160, y + 65)
            if i:
                page.insert_image(rect, stream=b"P6\n16 8\n255\n" + b"\0\0\xff" * 128,
                                  keep_proportion=False)
            else:
                page.draw_rect(rect, fill=(0, 1, 0))
                if case == "mixed":
                    page.insert_image(Rect(x + 20, y + 20, x + 30, y + 30),
                                      stream=b"P6\n1 1\n255\n\xff\0\0")
            letter = {"restart": "a", "missing-letter": "c"}.get(case, "b") if i else "a"
            page.insert_text((x + 5, y + 18), f"({letter})")
            if i or case == "already-claimed":
                page.insert_text((72, y + 88), f"Figure {i + 1}. Panel data.")
            if (not i and case == "text-after") or (i and case == "text-before"):
                page.insert_text((72, 60 if i else 385), "Intervening text.")
            if not i and case == "unlabelled-plot":
                page.draw_rect(Rect(80, 390, 220, 475), fill=(1, 0, 0))
            if not i and rotate_first:
                page.set_rotation(90)
        return bytes(doc.tobytes())


@pytest.mark.parametrize("case", ["restart", "missing-letter", "already-claimed", "text-after",
                                  "text-before", "unlabelled-plot", "misaligned"])
def test_ambiguous_panel_sequences_are_not_joined(case: str) -> None:
    with ArticlePDF(_panel_pdf(case), identifier="panels") as article:
        assert article.figures["2"].preceding_parts == ()


def test_component_getters_visit_all_parts_in_page_order() -> None:
    """Selecting by the anchor page returns components on both physical pages."""
    with ArticlePDF(_panel_pdf("mixed"), identifier="panels") as article:
        drawings = article.get_figure_drawings("2", page_index=1)
        assert len(drawings) == 1 and drawings[0] is article._drawings_cache[0][0]
        images = article.get_figure_images("2", page_index=1)
        assert len(images) == 2
        assert all(im is article._images_cache[n][0] for n, im in enumerate(images))
        copies = article.get_figure_images("2", copy_objects=True)
        assert copies == images and all(a is not b for a, b in zip(copies, images, strict=True))
        assert [b.text for b in article.get_figure_texts("2")] == ["(a)", "(b)"]
        with pytest.raises(KeyError):
            article.get_figure_drawings("2", page_index=0)


def test_reassigned_parts_are_shared_by_marking_and_export(tmp_path: Path) -> None:
    """Reassign (not just mutate) cached bounds, then check both public outputs."""
    with ArticlePDF(_panel_pdf("pair"), identifier="panels") as article:
        f = article.figures["2"]
        previous = Rect(100, 240, 140, 260)
        f.preceding_parts = ((0, previous),)
        f.rect = Rect(110, 100, 130, 115)
        assert article.get_figure_rects(0, copy_rects=False)[0] is previous
        assert article.get_figure_rects(1, copy_rects=False)[0] is f.rect
        marked = article.mark("figures", output_path=tmp_path, stroke_color=(1, 0, 1))
        png = Pixmap(str(article.extract_figures("2", output_path=tmp_path, dpi=72)["2"]))
        assert (png.width, png.height) == (40, 20 + 6 + 15)
        with pymupdf.open(marked) as doc:
            for page, expected in zip(doc, (previous, f.rect), strict=True):
                assert [d["rect"] for d in page.get_drawings() if d["color"] == (1, 0, 1)] == [expected]


def test_scoped_caption_lookup_never_enumerates_unrelated_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """After caption discovery only the requested page and its predecessor are relevant."""
    with ArticlePDF(_panel_pdf("pair"), identifier="panels") as article:
        for kind in ("figure", "table"):
            key = f"_{kind}_captions_cache"
            captions = getattr(article, key)
            local = MagicMock(wraps=captions)
            local.__getitem__.side_effect = captions.__getitem__
            local.values.side_effect = AssertionError("scanned all caption pages")
            monkeypatch.setitem(article.__dict__, key, local)
            getter = article.get_figure_caption_rects if kind == "figure" else article.get_table_caption_rects
            getter(1)
            local.__getitem__.assert_called_once_with(1)
            local.get.assert_called_once_with(0, ())


def test_split_export_reuses_layout_and_releases_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No previous crop or saved canvas remains live when the next crop is allocated."""
    sources: list[ReferenceType[Pixmap]] = []
    saved: list[ReferenceType[Pixmap]] = []
    render = Page.get_pixmap
    save = Pixmap.save

    def render_tracked(page: Page, *, dpi: int, clip: Rect) -> Pixmap:
        assert all(r() is None for r in sources), "previous crop retained"
        assert all(r() is None for r in saved), "previous canvas retained"
        pixels = render(page, dpi=dpi, clip=clip)
        sources.append(ref(pixels))
        return pixels

    def save_tracked(pixels: Pixmap, path: Path, *, output: str) -> None:
        saved.append(ref(pixels))
        save(pixels, path, output=output)

    with ArticlePDF(_panel_pdf("pair"), identifier="panels") as article:
        assert article.figures["2"].preceding_parts
        monkeypatch.setattr(Page, "get_pixmap", render_tracked)
        monkeypatch.setattr(Pixmap, "save", save_tracked)
        with patch.object(article, "_render_crop", wraps=article._render_crop) as crop, \
             patch.object(article, "_stack_layout", wraps=article._stack_layout) as layout:
            article.extract_figures("2", output_path=tmp_path, extract_bases=True, dpi=72)
            assert crop.call_count == 2  # once per fragment, not per layer
            layout.assert_called_once()
    assert len(sources) == 6 and len(saved) == 4
    assert all(r() is None for r in sources + saved)


def test_auto_dpi_rotation_and_blank_component_layers(tmp_path: Path) -> None:
    """Common DPI comes from the earlier vector page; layers keep the same slots."""
    with ArticlePDF(_panel_pdf("pair", rotate_first=True), identifier="panels") as article:
        png = article.extract_figures("2", output_path=tmp_path, extract_bases=True)["2"]
        figure = article.figures["2"]
        (n, rect), = figure.preceding_parts
        first = article.file[n].get_pixmap(dpi=300, clip=rect * article.file[n].rotation_matrix)
        last = article.file[1].get_pixmap(dpi=300, clip=figure.rect)
    full = Pixmap(str(png))
    assert (full.width, full.height) == (last.width, first.height + 25 + last.height)
    top = (full.width // 2, first.height // 2)
    bottom = (full.width // 2, full.height - last.height // 2)
    for suffix, colors in [("", [(0, 255, 0), (0, 0, 255)]),
                           ("_vectors", [(0, 255, 0), (255, 255, 255)]),
                           ("_image", [(255, 255, 255), (0, 0, 255)])]:
        pix = Pixmap(str(tmp_path / f"panels_fig_2{suffix}.png"))
        assert (pix.width, pix.height) == (full.width, full.height)
        assert [pix.pixel(*point) for point in (top, bottom)] == colors
