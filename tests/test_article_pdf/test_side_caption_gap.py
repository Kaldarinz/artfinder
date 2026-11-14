"""Side captions reach the figure set across the blank space of their row."""

from collections.abc import Iterator

import pymupdf
import pytest
from pymupdf import Rect

from artfinder.article_pdf import ArticlePDF

FIGURE = Rect(300, 70, 440, 200)
CAPTION = Rect(40, 80, 140, 110)
PROSE = (
    "The colloidal suspension was characterized by optical spectroscopy. "
    "Absorbance was recorded from 200 to 1300 nm."
)


def _side_caption_pdf(figure: Rect) -> bytes:
    """
    Build a page whose caption is parked in a column left of its figure.

    Parameters
    ----------
    figure : Rect
        Rectangle of the figure drawn on the page.

    Returns
    -------
    bytes
        PDF with running text, a narrow caption column and one figure.
    """

    with pymupdf.open() as document:
        document.set_metadata({"subject": "doi:10.1234/side-caption-gap"})
        page = document.new_page(width=480, height=600)
        page.draw_rect(figure, color=(0, 0, 1), fill=(0.8, 0.8, 1))
        page.insert_text(
            (CAPTION.x0, CAPTION.y0 + 10),
            "Figure 1. Absorbance spectra.",
            fontname="hebo",
            fontsize=9,
        )
        for paragraph in range(5):
            for line in range(2):
                page.insert_text(
                    (40, 300 + 40 * paragraph + 12 * line), PROSE, fontsize=9
                )
        return document.tobytes()


@pytest.fixture
def article() -> Iterator[ArticlePDF]:
    """Provide a page laid out as BMC sets a side caption.

    Returns
    -------
    Iterator[ArticlePDF]
        Open one-page article.
    """

    with ArticlePDF(_side_caption_pdf(FIGURE), identifier="side-caption") as pdf:
        yield pdf


def test_side_caption_takes_the_figure_across_a_layout_gap(
    article: ArticlePDF,
) -> None:
    """The blank between a caption and the figure beside it is the layout's."""
    assert article.MAX_FIGURE_CAPTION_GAP < FIGURE.x0 - CAPTION.x1
    assert article.figures["1"].rect == FIGURE


@pytest.mark.parametrize(
    ("figure", "multiples"),
    [
        (FIGURE, "row"),
        (Rect(40, 160, 180, 290), "above"),
    ],
    ids=["beside", "below-the-caption"],
)
def test_caption_distance_limit(
    article: ArticlePDF, figure: Rect, multiples: str
) -> None:
    """Only a caption beside its figure is limited by the width of the row."""
    limit = article._max_caption_distance(figure, CAPTION)
    if multiples == "row":
        assert limit == pytest.approx(
            article.paragraph_width.mean * article.SIDE_CAPTION_WEIGHT
        )
    else:
        assert limit == pytest.approx(article.MAX_FIGURE_CAPTION_GAP)
