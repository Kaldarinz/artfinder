"""Caption leading, size filtering, and page-break regressions."""

from pathlib import Path

import pymupdf
import pytest

from artfinder.article_pdf import ArticlePDF
from artfinder.dataclasses import TextBlockPDF


@pytest.mark.parametrize("kind", ["Figure", "Table"])
@pytest.mark.parametrize("pitch", [18.0, 22.0])
@pytest.mark.parametrize("line_count", [2, 3])
def test_caption_font_has_its_own_leading(
    tmp_path: Path, kind: str, pitch: float, line_count: int
) -> None:
    """Learn intermediate caption leading without absorbing body text."""
    source = tmp_path / "leading.pdf"
    with pymupdf.open() as document:
        for number in (1, 2):
            page = document.new_page()
            for row in range(10):
                page.insert_text(
                    (72, 60 + 28 * row),
                    "Body text uses wider leading and a larger size.",
                    fontname="tiro",
                    fontsize=12,
                )
            page.insert_text(
                (90, 400),
                f"{kind} {number}: Measurements of the sample",
                fontname="tiro",
                fontsize=10,
            )
            for row in range(1, line_count):
                page.insert_text(
                    (90, 400 + pitch * row),
                    "continued here." if row == line_count - 1 else "with more detail",
                    fontname="tiro",
                    fontsize=10,
                )
            page.insert_text(
                (90, 400 + pitch * (line_count - 1) + 28),
                "A following body paragraph must remain outside the caption.",
                fontname="tiro",
                fontsize=12,
            )
        document.save(source)

    with ArticlePDF(source) as article:
        assert article.line_pitch == pytest.approx(28)
        cache = (
            article._figure_captions_cache if kind == "Figure"
            else article._table_captions_cache
        )
        for page_no in (0, 1):
            (caption,) = cache[page_no]
            middle = " with more detail" if line_count == 3 else ""
            assert caption.text == f"Measurements of the sample{middle} continued here."
            assert caption.lines_no == line_count


@pytest.mark.parametrize("kind", ["Figure", "Table"])
def test_size_distinguishes_same_family_body_reference(
    tmp_path: Path, kind: str
) -> None:
    """Discard a duplicate body reference but retain a unique-size caption."""
    source = tmp_path / "sizes.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        for number, y in ((1, 100), (2, 200), (3, 300)):
            page.insert_text(
                (90, y),
                f"{kind} {number}: Actual caption {number}.",
                fontname="tiro",
                fontsize=12 if number == 3 else 10,
            )
        page.insert_text(
            (72, 500),
            f"{kind} 1, respectively, is discussed in this paragraph.",
            fontname="tiro",
            fontsize=12,
        )
        document.save(source)

    with ArticlePDF(source) as article:
        cache = (
            article._figure_captions_cache if kind == "Figure"
            else article._table_captions_cache
        )
        assert {c.label: c.text for c in cache[0]} == {
            str(n): f"Actual caption {n}." for n in (1, 2, 3)
        }


@pytest.mark.parametrize("kind", ["Figure", "Table"])
def test_caption_size_filter_tolerates_pdf_transform_variation(
    tmp_path: Path, kind: str
) -> None:
    """Count slightly different nominal sizes as one caption font."""
    source = tmp_path / "transforms.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        for number, size in enumerate((9.9, 10.0, 10.1, 10.2), 1):
            page.insert_text(
                (90, 60 + 60 * number),
                f"{kind} {number}: Actual caption {number}.",
                fontname="tiro",
                fontsize=size,
            )
        for number in (1, 2, 3):
            page.insert_text(
                (72, 350 + 60 * number),
                f"{kind} {number}, as discussed in the body.",
                fontname="tiro",
                fontsize=12,
            )
        document.save(source)

    with ArticlePDF(source) as article:
        cache = (
            article._figure_captions_cache if kind == "Figure"
            else article._table_captions_cache
        )
        assert {c.label: c.text for c in cache[0]} == {
            str(n): f"Actual caption {n}." for n in (1, 2, 3, 4)
        }


@pytest.mark.parametrize("kind", ["Figure", "Table"])
def test_local_leading_does_not_bridge_a_font_change(
    tmp_path: Path, kind: str
) -> None:
    """Corroborated caption leading does not admit a different-size paragraph."""
    source = tmp_path / "different-font.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (90, 100),
            f"{kind} 1: A single-line caption.",
            fontname="tiro",
            fontsize=10,
        )
        page.insert_text(
            (90, 118),
            "A separate paragraph with intermediate leading.",
            fontname="tiro",
            fontsize=12,
        )
        for y, text in (
            (200, f"{kind} 2: Another caption"),
            (218, "with more detail"),
            (236, "continued here."),
        ):
            page.insert_text((90, y), text, fontname="tiro", fontsize=10)
        document.save(source)

    with ArticlePDF(source) as article:
        # Model a word-processor export with a block for every line.
        article._text_cache[0] = tuple(
            TextBlockPDF(rect=line.rect, lines=[line])
            for block in article._text_cache[0]
            for line in block.lines
        )
        article.__dict__["line_pitch"] = 28.0
        cache = (
            article._figure_captions_cache if kind == "Figure"
            else article._table_captions_cache
        )
        assert cache[0][0].text == "A single-line caption."
        assert cache[0][1].text == "Another caption with more detail continued here."


@pytest.mark.parametrize("kind", ["Figure", "Table"])
@pytest.mark.parametrize(
    "obstruction", ["none", "font", "indent", "finished", "opening", "gap"]
)
def test_caption_continues_across_page_break(
    tmp_path: Path, kind: str, obstruction: str
) -> None:
    """Mark only matching continuations, never unrelated next-page paragraphs."""
    source = tmp_path / "page-break.pdf"
    with pymupdf.open() as document:
        first = document.new_page()
        first.insert_text(
            (90, 730),
            f"{kind} 1: Measurements and optical properties",
            fontname="tiro",
            fontsize=10,
        )
        first.insert_text(
            (90, 748),
            "structure." if obstruction == "finished"
            else "structure and temperature dependence measured",
            fontname="tiro",
            fontsize=10,
        )
        second = document.new_page()
        second.insert_text(
            (110 if obstruction == "indent" else 90, 70),
            f"{kind.lower()} 2: Another caption." if obstruction == "opening"
            else "their measured dependence on temperature",
            fontname="tiro",
            fontsize=12 if obstruction == "font" else 10,
        )
        second.insert_text(
            (90, 110 if obstruction == "gap" else 88),
            "and composition.",
            fontname="tiro",
            fontsize=10,
        )
        second.insert_text(
            (90, 150),
            "A separate body paragraph stays outside the caption.",
            fontname="tiro",
            fontsize=12,
        )
        document.save(source)

    with ArticlePDF(source) as article:
        # Isolate page-break recognition from the sparse document's median.
        article.__dict__["line_pitch"] = 18.0
        cache = (
            article._figure_captions_cache if kind == "Figure"
            else article._table_captions_cache
        )
        caption = cache[0][0]
        if obstruction in {"none", "gap"}:
            assert "their measured dependence on temperature" in caption.text
            assert caption.text.endswith(
                "temperature" if obstruction == "gap" else "composition."
            )
            assert len(caption.continuations) == 1
            page_no, rect = caption.continuations[0]
            assert page_no == 1
            assert rect.y1 < 100
            getter = (
                article.get_figure_caption_rects if kind == "Figure"
                else article.get_table_caption_rects
            )
            assert rect in getter(1)
            assert getter(article.file[1]) == [rect]
            assert rect not in getter(0)
            assert rect not in getter(1, clip=pymupdf.Rect(0, 200, 500, 400))
            assert len(getter()) == 2
            copied = getter(1)[0]
            assert copied is not rect
            copied.x0 -= 10
            assert getter(1, copy_rects=False)[0] is rect
            element = "figure_captions" if kind == "Figure" else "table_captions"
            marked = article.mark(element, output_path=tmp_path / "marked")
        else:
            assert caption.continuations == ()
            assert "their measured" not in caption.text
            return

    with pymupdf.open(marked) as document:
        for page in document:
            marks = [
                drawing for drawing in page.get_drawings()
                if drawing["color"] == (1, 0, 0)
            ]
            assert len(marks) == 1


CORPUS_PDF = (
    Path(__file__).parent / "article_pdfs"
    / "nanostructuring of alcu alloy nps obtained by flasis.pdf"
)


def test_alcu_captions_are_complete_and_body_reference_is_not_marked() -> None:
    """Keep all caption lines, including Figure 12's next-page continuation."""
    with ArticlePDF(CORPUS_PDF) as article:
        captions = [c for values in article._figure_captions_cache.values() for c in values]
        assert len(captions) == article.figure_count == 14
        assert all(c.lines_no >= 2 for c in captions)
        assert article.get_figure_caption_rects(24) == []
        assert len(article.get_figure_caption_rects(28)) == 1
        assert article.figure_captions["12"].endswith("a modal radius of 10.1 nm.")
        assert article._figure_label_to_page_ind["8"] == 19
