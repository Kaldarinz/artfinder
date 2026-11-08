"""Keep already captioned figures out of the previous-page carry-over region."""

import json
from pathlib import Path

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF


TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"
CAPTION_ABOVE_PDF = (
    "comparative in vivo assessment of some adverse bioeffects of equidimensional"
    " gold and silver nanoparticles and the attenuation of nanosilver’s effects"
    " with a complex of innocuous bioprotectors.pdf"
)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_caption_above_does_not_steal_previous_figure(reverse: bool) -> None:
    """Keep consecutive caption-above figures on their own pages in either order."""
    rects = [Rect(100, 105, 400, 350), Rect(120, 110, 420, 360)]
    with pymupdf.open() as document:
        for number, rect in enumerate(rects, start=1):
            page = document.new_page()
            page.insert_text(
                (100, 80),
                f"Figure {number}. Measurements of sample {number}.",
                fontname="hebo",
                fontsize=10,
            )
            page.draw_rect(rect, color=(0, 0, 1), fill=(0.8, 0.8, 1))
        data = document.tobytes()
    with ArticlePDF(data, identifier="caption-above") as article:
        for page_no in ([1, 0] if reverse else [0, 1]):
            figures = article.get_figures(page_no)
            label = str(page_no + 1)
            assert set(figures) == {label}
            assert figures[label].rect == rects[page_no]
            assert article._figure_label_to_page_ind[label] == page_no


@pytest.mark.parametrize("preceding_figure", [False, True])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_unclaimed_graphics_can_still_carry_over(
    preceding_figure: bool, reverse: bool
) -> None:
    """Carry an unclaimed image over even below another already matched figure."""
    expected = Rect(100, 450, 400, 650)
    with pymupdf.open() as document:
        page = document.new_page()
        if preceding_figure:
            page.insert_text(
                (100, 80), "Figure 1. First sample.", fontname="hebo", fontsize=10
            )
            page.draw_rect(
                Rect(100, 105, 400, 300), color=(0, 0, 1), fill=(0.8, 0.8, 1)
            )
        page.draw_rect(expected, color=(0, 0, 1), fill=(0.8, 0.8, 1))
        page = document.new_page()
        page.insert_text(
            (100, 80), "Figure 2. Second sample.", fontname="hebo", fontsize=10
        )
        data = document.tobytes()
    with ArticlePDF(data, identifier="genuine-carryover") as article:
        figures_by_page = {
            page_no: article.get_figures(page_no)
            for page_no in ([1, 0] if reverse else [0, 1])
        }
        assert not figures_by_page[1]
        assert set(figures_by_page[0]) == ({"1", "2"} if preceding_figure else {"2"})
        assert figures_by_page[0]["2"].rect == expected
        assert article._figure_label_to_page_ind["2"] == 0


def test_caption_above_article_page_local_goldens() -> None:
    """Check every page, including both parts hidden by the label-keyed API."""
    expected_rects: dict[str, list[float]] = json.loads(
        (TEST_PDFS_DIR / "figure_rects.json").read_text(encoding="utf-8")
    )[CAPTION_ABOVE_PDF]
    expected_captions: dict[str, str] = json.loads(
        (TEST_PDFS_DIR / "all_fig_captions.json").read_text(encoding="utf-8")
    )[CAPTION_ABOVE_PDF]
    labels_by_page = {
        7: ["1"],
        8: ["1", "2"],
        10: ["3"],
        11: ["4", "5"],
        16: ["6"],
        17: ["7", "8"],
        23: ["9"],
        24: ["10"],
    }
    with ArticlePDF(TEST_PDFS_DIR / CAPTION_ABOVE_PDF) as article:
        for page_no in range(len(article.file)):
            figures = article.get_figures(page_no)
            assert list(figures) == labels_by_page.get(page_no, [])
            for label, figure in figures.items():
                if page_no == 7:
                    # Figure 1 spans pages 8–9. The public dictionary retains
                    # the last part ("Cont."); keep the first part covered too.
                    rect = [
                        130.50003051757812,
                        423.3599853515625,
                        464.6999816894531,
                        726.1316528320312,
                    ]
                    caption = (
                        "Typical cell surface topography imaged by semi-contact atomic"
                        " force microscopy (AFM), for various groups of samples: (a)"
                        " and (b) controls; (c) and (d) after instillation of 50 nm gold;"
                        " (e) and (f) 49 nm silver; (g) and (h) 1 μm silver. Scan sizes:"
                        " (a), (c), (e), (g) 10 × 10 μm 2; (b), (d), (f), (h) 2 × 2 μm 2."
                    )
                else:
                    rect = expected_rects[label]
                    caption = expected_captions[label]
                assert tuple(figure.rect) == pytest.approx(tuple(rect))
                assert figure.caption.text == caption
                assert figure.rect.y0 > figure.caption.rect.y1
