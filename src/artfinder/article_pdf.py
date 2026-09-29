"""
Article class for PDF analysis and figure extraction.

This module provides a comprehensive class for analyzing scientific articles in PDF format,
extracting figures, captions, and analyzing document structure.

A figure is identified by its caption label rather than by a number: `"1"` for
`Figure 1`, `"S1"` for `Figure S1`, `Fig. S1` or `Supplementary Figure 1`. This
keeps main-text and supplementary figures apart in a document that contains
both, such as a preprint with its supplementary information appended.
"""

import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from copy import copy, deepcopy
from dataclasses import replace
from functools import cached_property
from itertools import chain
from math import floor, inf
from statistics import median
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from warnings import warn

import numpy as np
import pandas as pd
import pymupdf
from pymupdf import Pixmap, Rect
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]

if TYPE_CHECKING:
    # PyMuPDF binds `Page` to a string before it defines the class, so mypy never
    # sees the class, nor what `Document` returns for a page.
    class Page(Any): ...

else:
    from pymupdf import Page

from artfinder.dataclasses import (
    DocumentElementsPDF,
    DrawingObjectPDF,
    FigureCaptionPDF,
    FigureClusterPDF,
    FigurePDF,
    ImageInfoPDF,
    KeyedDict,
    Size,
    TablePDF,
    TextBlockPDF,
    TextLinePDF,
)
from artfinder.helpers import (
    clip_to_grid,
)

logger = logging.getLogger(__name__)


class ArticlePDF:
    """
    A class for analyzing scientific articles in PDF format.

    This class provides methods for extracting figures, captions, analyzing document
    structure (columns, paragraphs), and identifying document headers.

    Figures are composed of vector graphics (drawings) and raster images and text elements.
    Every figure is keyed by its label — the figure number as written in the
    caption, prefixed with `S` for a supplementary figure, e.g. `1` or `S1`.

    Known limitations
    -----------------
    The caption filter in `_find_figure_captions` keeps only captions whose font
    matches the single most common caption font in the whole document. In a
    combined main-text-plus-supplementary PDF whose supplementary captions are
    set in a different font from the main ones, the smaller of the two sets is
    therefore dropped.

    Figure labels that no test PDF has needed yet are not recognised: `Fig. SI1`,
    `Figure S-1`, and Roman numerals.
    """

    COLUMN_NO_FITTING_FACTOR = 1.1
    "Factor to adjust column fitting when estimating number of columns."
    HEADER_MAX_FRACTION = 0.09
    "Maximum fraction of page height to consider for header detection."
    FOOTER_MAX_FRACTION = 0.09
    "Maximum fraction of page height to consider for footer detection."
    RECTS_CLIP_PRECISION = 0
    "Precidion digits for clipping rects coordinates."
    MAX_IMAGE_AREA = 0.8
    "Maximum fraction of page area for an image to be considered valid."
    POINTS_PER_INCH = 72
    "Points in an inch, the unit PDF coordinates are given in."
    MIN_FIGURE_DPI = 150
    "Lowest resolution `dpi='auto'` rasterizes a figure at."
    MAX_FIGURE_DPI = 300
    """Highest resolution `dpi='auto'` rasterizes a figure at. Caps both the
    output size and the pixmap area for a figure carrying a print-resolution
    image."""
    VECTOR_FIGURE_DPI = 300
    """Resolution `dpi='auto'` rasterizes a figure holding no raster image at.
    Vectors and text carry no resolution of their own to measure, and render
    the more faithfully the higher it is."""
    MARGIN = 2
    "Margin in points for rectangles."
    MAX_LINE_PITCH = 60.0
    """Largest distance between consecutive text lines, in points, still counted
    as ordinary leading when measuring `line_pitch`."""
    MIN_PARAGRAPH_OVERLAP = 0.8
    """Fraction of a trailing line that must sit within the horizontal span of a
    paragraph for the line to belong to it."""
    LINE_PITCH_TOLERANCE = 0.15
    """How far the distance between two lines may differ from `line_pitch` or
    `paragraph_pitch`, as a fraction of it, for them to still be consecutive lines
    of one caption or paragraph."""
    MAX_SINGLE_SPACING = 1.3
    """Largest distance between consecutive lines of a single-spaced caption, as
    a multiple of the height of a line. Line boxes are shorter than the leading
    they are set at: 12 pt Times single-spaced by a word processor puts lines of
    13.5 pt 16 pt apart."""
    MIN_FIGURE_IMAGE_OVERLAP = 0.9
    """Fraction of an image reaching above the bounds of a figure that must lie
    within them for the image to be part of the figure."""
    FIGURE_GRAPHICS_GAP = 10.0
    """Widest blank space, in points, between two drawings or images of one
    cluster of graphics."""
    FIGURE_LABEL_GAP = 15.0
    """Widest blank space, in points, between a cluster of graphics and a label
    set beside it. Wider than `FIGURE_GRAPHICS_GAP`: a rotated axis title stands
    off from the tick labels by more than the graphics stand off each other."""
    MIN_FIGURE_SIDE = 25.0
    """Shortest a cluster of graphics may be on its longer side, in points, to
    be a figure rather than a logo, an icon or a stroke of an equation."""
    MAX_FIGURE_CAPTION_GAP = 120.0
    """Widest blank space, in points, between a figure and its caption. Wide
    enough for a side caption set flush with the page edge."""
    MAX_FIGURE_PANEL_GAP = 40.0
    """Widest blank space, in points, across which a figure takes in a cluster
    of graphics no caption claimed — a panel set apart from the others."""
    CAPTION_ABOVE_FIGURE_WEIGHT = 1.5
    """Factor on the distance from a caption to a figure below it: a caption
    sits under its figure more often than over it."""
    SIDE_CAPTION_WEIGHT = 1.2
    "Factor on the distance from a caption to a figure beside it."
    MAX_HAIRLINE_WIDTH = 2.0
    """Thickest a drawing may be, in points, to be a rule rather than an area."""
    MIN_PAGE_RULE_LENGTH = 0.4
    """Shortest a hairline may be, as a fraction of the page width, to be a
    rule of the page — under a header, over the footnotes — rather than of a
    figure."""
    MAX_CAPTION_FIGURE_OVERLAP = 4.0
    """How far, in points, a figure may reach past the edge of the caption it
    faces and still count as above or below it."""
    MAX_FIGURE_TEXT_OVERLAP = 0.2
    """Largest fraction of a paragraph or caption a figure may cover when it
    takes in a panel no caption claimed."""
    WHITE_LEVEL = 0.97
    """Lowest value every component of a colour must reach for it to be taken
    as white, which leaves a drawing painted in it invisible on the page."""
    DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
    "Pattern of a DOI in text. The prefix dot is literal: `1000/x` is not a DOI."
    CAPTION_PATTERN = re.compile(
        r"^\s*(?:(?P<supp>supplementary|supplemental|supporting)\s+)?"
        r"Fig(?:\.|ure\.?|)\s+(?:(?P<prefix>S)\s?)?(?P<number>\d+)(?P<suffix>S)?(?!\w)[^\w(\[]*",
        re.IGNORECASE,
    )
    """Pattern of a figure caption opening, e.g. `Fig. 1`, `Figure S2.`,
    `Figure 2S.` or `Supplementary Figure 3`. The `supp`, `prefix` and `suffix`
    groups mark a supplementary figure, the `number` group holds its digits.
    A label run into a letter is not an opening: `Figure 1a shows` starts a
    paragraph referring to a panel, set in the same font as a caption."""
    TABLE_CAPTION_PATTERN = re.compile(
        r"^\s*(?:(?P<supp>supplementary|supplemental|supporting)\s+)?"
        r"Table\s+(?:(?P<prefix>S)\s?)?(?P<number>\d+)(?P<suffix>S)?(?!\w)[^\w(\[]*",
        re.IGNORECASE,
    )
    """Pattern of a table caption opening, e.g. `Table 1.` or `Table S2.`, with
    the groups of `CAPTION_PATTERN`."""
    MAX_CAPTION_DISTANCE = 1.5
    """How far from its caption a table body may start, and how far its blocks
    may follow each other, in line pitches."""
    MIN_TABLE_RULING_OVERLAP = 0.5
    """Fraction of a drawing that must lie inside a table body for the drawing to
    be part of the table — its ruling or the shading of its header."""
    MAX_TABLE_CELL_CHARS = 30
    """Longest median line, in characters, for a block to read as table cells
    rather than as running text."""
    FONT_STYLE_PATTERN = re.compile(r"(?:ps)?(?:mt)?(?:[-,].*)?$")
    """Pattern of what follows a font's family in its name: a style after a dash
    or a comma (`Arial-BoldMT`, `Arial,Italic`) or a vendor suffix
    (`TimesNewRomanPSMT`). Matched against the lowercased name."""
    FONT_SIZE_TOLERANCE = 0.5
    "How far, in points, a font size may fall below the body size and still count as it."
    LINE_NUMBER_PATTERN = re.compile(r"^\d{1,4}$")
    """Pattern of a manuscript line number: a bare integer and nothing else on
    the line."""
    LINE_NUMBER_MAX_WIDTH = 0.1
    "Maximum width of a line number, as a fraction of the page width."
    LINE_NUMBER_ALIGN_TOLERANCE = 4.0
    "Maximum spread, in points, of the aligned edge of a line number column."
    LINE_NUMBER_MIN_COUNT = 15
    "Minimum number of members of a line number column."
    LINE_NUMBER_MULTILINE_PAGE_FRACTION = 0.25
    """Minimum fraction of pages that must carry two or more numbers of the same
    column. Separates line numbering from a page number in the footer, which
    occurs once per page."""
    TILE_MIN_REPEATED_GLYPHS = 10
    """Minimum number of glyphs drawn twice across a tile seam for a page to be
    read as tiled. No untiled page repeats a single one."""
    TILE_SEAM_TOLERANCE = 1.0
    """How far, in points, a fragment may start past the end of the fragment it
    continues, or its baseline may differ, and still be one line with it; and how
    far it must reach back over it to repeat its last glyph."""
    TILE_MAX_WORD_SPACE = 0.5
    """Widest gap between a line and the fragment continuing it, as a fraction of
    the line's height: a seam can fall on a word space."""
    TILE_MIN_WORD_SPACE = 0.15
    """Narrowest gap between a line and the fragment continuing it, as a fraction
    of the font size, read as a word space when neither of them carries one."""
    XMP_DOI_PATTERN = re.compile(
        r"(?:prism:doi|dc:identifier|doi)[^>]*>\s*(?:doi:)?(10\.\d{4,9}/[^<\s]+)",
        re.IGNORECASE,
    )
    "Pattern of a DOI inside the XMP metadata packet."
    REFERENCES_HEADING_PATTERN = re.compile(
        r"^[ \t]*(?:\d+\.?[ \t]*)?"
        r"(?:references and notes|references|bibliography|literature cited)"
        r"[ \t]*:?[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    "Pattern of a standalone bibliography heading, used to stop the DOI search."

    def __init__(self, pdf: PathLike | str | bytes, identifier: str = ""):
        """
        Initialize Article with a PDF file.

        Parameters
        ----------
        pdf : PathLike | str | bytes
            Path to the PDF file to analyze, or its raw bytes.
        identifier : str, optional
            Identifier for the article. If empty, falls back to the filename,
            then the DOI, then the first 50 characters of the first page's text.

        Raises
        ------
        FileNotFoundError
            If the PDF file does not exist.
        ValueError
            If the file cannot be opened as a PDF.
        """

        self._raw_text_cache: dict[int, tuple[TextBlockPDF, ...]] = KeyedDict(
            self._get_raw_text_blocks_from_page
        )
        """Text blocks keyed by page number, before line numbers are removed."""
        self._text_cache: dict[int, tuple[TextBlockPDF, ...]] = KeyedDict(
            self._get_text_blocks_from_page
        )
        """Text blocks keyed by page number."""
        self._drawings_cache: dict[int, tuple[DrawingObjectPDF, ...]] = KeyedDict(
            self._get_drawing_objs_from_page
        )
        """Drawing objects keyed by page number."""
        self._images_cache: dict[int, tuple[ImageInfoPDF, ...]] = KeyedDict(
            self._get_image_info_from_page
        )
        self._paragraphs_cache: dict[int, tuple[Rect, ...]] = KeyedDict(
            self._get_paragraphs_from_page
        )
        """Paragraph rectangles keyed by page number."""
        self._figure_clusters_cache: dict[int, tuple[FigureClusterPDF, ...]] = (
            KeyedDict(self._get_figure_clusters_from_page)
        )
        """Clusters of graphics that could make up figures, keyed by page number."""
        self._figure_matches_cache: dict[int, dict[int, Rect]] = KeyedDict(
            self._match_figures_on_page
        )
        """Figure rectangles found from their graphics, keyed by page number,
        then by the index of their caption among the page's figure captions."""
        self._figures_cache: dict[int, tuple[FigurePDF, ...]] = KeyedDict(
            self._get_figures_from_page
        )
        """Figures keyed by page number."""
        self._tables_cache: dict[int, tuple[TablePDF, ...]] = KeyedDict(
            self._get_tables_from_page
        )
        """Tables keyed by page number."""
        self._table_label_to_page_ind: dict[str, int] = {}
        """Page index of each table, keyed by table label."""
        self._figure_label_to_page_ind: dict[str, int] = {}
        """Page index of each figure, keyed by figure label."""

        self.identifier = identifier
        if isinstance(pdf, (PathLike, str)):
            self.path = Path(pdf)
            if not self.path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf}")
            if len(self.identifier) == 0:
                self.identifier = self.path.name
        try:
            if hasattr(self, "path"):
                self.file = pymupdf.open(str(self.path))
            else:
                self.file = pymupdf.open(stream=pdf, filetype="pdf")
        except (
            Exception
        ) as e:  # noqa: BLE001 - pymupdf raises varied types for a bad file
            raise ValueError(f"Failed to open PDF file: {e}")

        if len(self.identifier) == 0:
            if self.doi is not None:
                self.identifier = self.make_valid_filename(self.doi)
            else:
                self.identifier = self.make_valid_filename(self._page(0).get_text()[:50])

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the PDF."""
        self.close()

    def close(self):
        """Close the PDF document."""
        if hasattr(self, "file"):
            self.file.close()

    def __repr__(self):
        return f"Article('{self.identifier}', pages={len(self.file)})"

    def _page(self, page_no: int) -> Page:
        """Page `page_no` of the document, typed as `Page` (see its import)."""
        return self.file[page_no]

    @staticmethod
    def make_valid_filename(name: str) -> str:
        """
        Make a valid filename from a given string.

        Parameters
        ----------
        name : str
            Input string to convert to a valid filename.

        Returns
        -------
        str
            Valid filename string.
        """

        # Replace invalid characters with underscores
        valid_name = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", name)
        # Truncate to a reasonable length
        return valid_name[:255]

    # Cached properties for expensive computations
    @cached_property
    def doi(self) -> str | None:
        """
        DOI of the article.

        Returns
        -------
        str | None
            DOI string if found, otherwise None.
        """

        return self._get_doi()

    @cached_property
    def paragraph_width(self) -> Size:
        """
        Paragraph width for the entire document.

        Returns
        -------
        Size
            Mean, min, max of paragraph widths.
        """

        return self._calc_paragraph_width()

    @cached_property
    def line_pitch(self) -> float:
        """
        Distance between the baselines of consecutive text lines.

        Measured as the median distance between consecutive lines, so it
        reflects the leading the document is actually set with. This is not the
        same as the height of a line: a manuscript set with extra leading spaces
        its lines further apart than their glyphs are tall, and a caption
        continued on the next line is then not one line height below its
        predecessor but one pitch.

        Lines are paired inside a text block and also across consecutive
        blocks that overlap horizontally. The second is what measures a
        document exported from a word processor, which puts nearly every line
        in a block of its own: counting only within blocks leaves it a sample
        or two — a title over its subtitle — and a pitch far wider than the
        leading, at which no caption's continuation line is recognized.

        Returns
        -------
        float
            Median distance between consecutive lines, in points. Falls back to
            the median line height if the document has no two such lines.
        """

        gaps: list[float] = []
        heights: list[float] = []

        def add_gap(line: TextLinePDF, next_line: TextLinePDF) -> None:
            gap = next_line.rect.y1 - line.rect.y1
            if 0 < gap < self.MAX_LINE_PITCH:
                gaps.append(gap)

        for page_no in range(self.file.page_count):
            blocks = self._text_cache[page_no]
            for block in blocks:
                heights.extend(line.rect.height for line in block.lines)
                for line, next_line in zip(block.lines, block.lines[1:]):
                    add_gap(line, next_line)
            for block, next_block in zip(blocks, blocks[1:]):
                last, first = block.lines[-1].rect, next_block.lines[0].rect
                if min(last.x1, first.x1) > max(last.x0, first.x0):
                    add_gap(block.lines[-1], next_block.lines[0])
        if gaps:
            return median(gaps)
        return median(heights) if heights else 0.0

    @cached_property
    def paragraph_pitch(self) -> float:
        """
        Distance between the baselines of consecutive lines of body paragraphs.

        `line_pitch` is measured over all the text of a document, and in a
        manuscript whose body is set at one and a half or double spacing the
        single-spaced references, tables and affiliations outnumber it. The
        leading of the body then differs from `line_pitch`, and the last line
        of a paragraph, narrower than the body, is not recognized as following
        it. Measuring over the lines of paragraph blocks alone gives the body's
        own leading.

        Lines are paired inside a block and across consecutive blocks that
        overlap horizontally, as for `line_pitch`.

        Returns
        -------
        float
            Median distance between consecutive paragraph lines, in points.
            Falls back to `line_pitch` if the document has no two such lines.
        """

        gaps: list[float] = []
        for page_no in range(self.file.page_count):
            lines = [
                line
                for block in self._text_cache[page_no]
                if self._is_paragraph_block(block)
                for line in block.lines
            ]
            for line, next_line in zip(lines, lines[1:]):
                gap = next_line.rect.y1 - line.rect.y1
                overlap = min(line.rect.x1, next_line.rect.x1) - max(
                    line.rect.x0, next_line.rect.x0
                )
                if overlap > 0 and 0 < gap < self.MAX_LINE_PITCH:
                    gaps.append(gap)
        return median(gaps) if gaps else self.line_pitch

    @cached_property
    def body_font(self) -> tuple[str, float]:
        """
        Font family and size the most text of the document is set in.

        The family is the one `_font_family` returns, so the bold and italic
        cuts of the body typeface count towards it too.

        Returns
        -------
        tuple[str, float]
            Font family and size, rounded to a tenth of a point. An empty family
            and a zero size if the document holds no text.
        """

        chars: Counter[tuple[str, float]] = Counter()
        for page_no in range(self.file.page_count):
            for block in self._text_cache[page_no]:
                for line in block.lines:
                    for span in line.spans:
                        font = (self._font_family(span.font), round(span.size, 1))
                        chars[font] += len(span.text.strip())
        most_common = chars.most_common(1)
        return most_common[0][0] if most_common else ("", 0.0)

    @cached_property
    def columns_number(self) -> int:
        """
        Number of columns in the document.

        Raises
        ------
        ValueError
            If the document has no pages with a valid width.

        Returns
        -------
        int
            The estimated number of columns (1, 2, etc.).
        """
        all_widths = [self._page(i).rect.width for i in range(self.file.page_count)]
        if len(all_widths):
            page_width = sum(all_widths) / len(all_widths)
        else:
            raise ValueError("No pages with valid width.")
        if self.paragraph_width.mean <= 0:
            # Nothing to divide the page by: a document of captions and figures
            # with no body text to measure, as a supplementary file often is.
            logger.debug(f"Paragraph width unknown, assuming one column in {self}.")
            return 1
        return floor(
            page_width / (self.paragraph_width.mean * self.COLUMN_NO_FITTING_FACTOR)
        )

    @cached_property
    def header_min_pages(self) -> int:
        """
        Number of pages an element must appear on to count as a header.

        A running header is not always printed on every page: many journals
        omit it on the first page, and some print it on alternating pages only,
        which in a four-page article leaves just two. The threshold therefore
        follows the page count instead of being fixed, low enough to accept
        alternating headers and high enough to reject an element that repeats on
        a couple of pages by coincidence.

        Returns
        -------
        int
            Minimum number of distinct pages.
        """

        return max(2, (self.file.page_count - 1) // 2)

    def _repeating_rects_in(self, band: Rect, is_header: bool) -> list[Rect]:
        """
        Internal method to find what repeats at the same place inside a band.

        An element takes part if it reaches into the band at all, and its whole
        rectangle is what is ranked: a header is often set deeper than any fixed
        fraction of the page, and requiring the element to fit inside the band
        would keep the very lines the band should have covered out of it.

        What separates page furniture from content is the body text, not a size:
        a header ends above every paragraph of its page, a footer starts below
        them all. That keeps out an element that merely dips into the band —
        a caption at the foot of a page, or a sidebar running the height of it.

        Note this reads `paragraph_width`, which is why `_calc_paragraph_width`
        selects on prose rather than on the tables, headers and footers found so
        far: it would otherwise be circular.

        Parameters
        ----------
        band : Rect
            Band of the page to search.
        is_header : bool
            Whether the band is at the top of the page rather than the bottom.

        Returns
        -------
        list[Rect]
            Rectangles appearing on the most pages, empty if none repeats on
            `header_min_pages` pages.
        """

        clip_precision = self.RECTS_CLIP_PRECISION
        while True:
            pages_of_rect: dict[Rect, set[int]] = {}
            for page_no in range(self.file.page_count):
                paragraphs = self.get_paragraph_rects(
                    page_no=page_no, copy_rects=False
                )
                body_top = min((rect.y0 for rect in paragraphs), default=None)
                body_bottom = max((rect.y1 for rect in paragraphs), default=None)

                rects = chain(
                    self.get_text_rects(page_no=page_no, copy_rects=False),
                    self.get_drawing_rects(page_no=page_no, copy_rects=False),
                    self.get_image_rects(page_no=page_no, copy_rects=False),
                )
                for rect in rects:
                    if not Rect(rect).intersects(band):
                        continue
                    if is_header:
                        if body_top is not None and rect.y1 > body_top:
                            continue
                    elif body_bottom is not None and rect.y0 < body_bottom:
                        continue
                    clipped = clip_to_grid(rect, clip_precision)
                    pages_of_rect.setdefault(clipped, set()).add(page_no)

            page_counts = {
                len(pages)
                for pages in pages_of_rect.values()
                if len(pages) >= self.header_min_pages
            }
            if len(page_counts):
                break
            clip_precision -= 1
            if clip_precision < -1:
                return []

        most_pages = max(page_counts)
        return [
            rect for rect, pages in pages_of_rect.items() if len(pages) == most_pages
        ]

    @cached_property
    def header_rect(self) -> Rect:
        """
        Document header rectangle.

        The header is whatever occupies the same place in the top band of at
        least `header_min_pages` distinct pages.

        Returns
        -------
        Rect
            The header rectangle if found, otherwise an empty Rect.
        """

        # We assume that all pages in the pdf file have the same size
        search_rect = cast(Rect, self._page(0).rect)
        search_rect.y1 = search_rect.height * self.HEADER_MAX_FRACTION

        common = self._repeating_rects_in(search_rect, is_header=True)
        if not common:
            # A document need not have a running header at all:
            # supplementary information and preprints usually do not.
            logger.debug(f"Header rectangle not found for {self}.")
            return Rect()
        search_rect.y1 = max(rect.y1 for rect in common)
        return search_rect

    @cached_property
    def footer_rect(self) -> Rect:
        """
        Document footer rectangle.

        The mirror of `header_rect`: whatever occupies the same place in the
        bottom band of at least `header_min_pages` distinct pages. It gives the
        elements that end at the bottom of a page — a table printed below its
        caption, a figure taking in its panels — something to stop against.

        Returns
        -------
        Rect
            The footer rectangle if found, otherwise an empty Rect.
        """

        page_rect = cast(Rect, self._page(0).rect)
        search_rect = Rect(
            page_rect.x0,
            page_rect.y1 - page_rect.height * self.FOOTER_MAX_FRACTION,
            page_rect.x1,
            page_rect.y1,
        )

        common = self._repeating_rects_in(search_rect, is_header=False)
        if not common:
            logger.debug(f"Footer rectangle not found for {self}.")
            return Rect()
        search_rect.y0 = min(rect.y0 for rect in common)
        return search_rect

    @cached_property
    def _running_matter_rects(self) -> frozenset[tuple[float, float, float, float]]:
        """
        Rectangles of the page furniture: running heads, footers, page numbers.

        Anything printed at the same place on as many pages as a header must
        appear on is furniture rather than content, wherever on the page it sits.

        Returns
        -------
        frozenset[tuple[float, float, float, float]]
            Coordinates of the repeating rectangles.
        """

        pages_of_rect: dict[Rect, set[int]] = {}
        for page_no in range(self.file.page_count):
            for rect in self.get_text_rects(page_no=page_no, copy_rects=False):
                clipped = clip_to_grid(rect, self.RECTS_CLIP_PRECISION)
                pages_of_rect.setdefault(clipped, set()).add(page_no)
        return frozenset(
            self._rect_key(rect)
            for rect, pages in pages_of_rect.items()
            if len(pages) >= self.header_min_pages
        )

    def _is_running_matter(self, rect: Rect) -> bool:
        """
        Internal method to tell whether a rectangle is page furniture.

        Parameters
        ----------
        rect : Rect
            Rectangle of a text block.

        Returns
        -------
        bool
            Whether the rectangle is among `_running_matter_rects`, compared on
            the grid those are rounded to.
        """

        return (
            self._rect_key(clip_to_grid(rect, self.RECTS_CLIP_PRECISION))
            in self._running_matter_rects
        )

    @cached_property
    def _line_number_rects(self) -> dict[int, frozenset[tuple[float, float, float, float]]]:
        """Rectangles of manuscript line numbers, keyed by page number."""

        return self._find_line_numbers()

    @cached_property
    def _figure_captions_cache(self) -> dict[int, tuple[FigureCaptionPDF, ...]]:
        """Figure captions in the document."""

        return self._find_figure_captions()

    @cached_property
    def figure_captions(self) -> dict[str, str]:
        """Figure captions in the document as text."""
        return {
            fig_label: figure.caption.text
            for fig_label, figure in self.figures.items()
        }

    @cached_property
    def figures(self) -> dict[str, FigurePDF]:
        return self.get_figures()

    @cached_property
    def _table_captions_cache(self) -> dict[int, tuple[FigureCaptionPDF, ...]]:
        """Table captions in the document."""

        return self._find_captions(self.TABLE_CAPTION_PATTERN)

    @cached_property
    def tables(self) -> dict[str, TablePDF]:
        """Tables in the document, keyed by table label."""

        return self.get_tables()

    @cached_property
    def table_captions(self) -> dict[str, str]:
        """Table captions in the document as text."""

        return {
            label: table.caption.text for label, table in self.tables.items()
        }

    @cached_property
    def figure_count(self) -> int:
        """Number of figures in the document."""
        return len(self.get_figure_caption_rects(copy_rects=False))

    # Public methods

    def get_text_rects(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        min_len: int | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:
        """
        Get rectangles for blocks of text from a specific page.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return text rectangles for all document
        clip : Rect | None, optional
            Rectangle to clip the text blocks.
        min_len : int | None, optional
            Minimum length of text blocks to include.
        copy_rects : bool, default=True
            Whether to return copies of the rectangles or references.

        Returns
        -------
        list[Rect]
            List of text rectangles.
        """

        if isinstance(page_no, Page):
            page_no = cast(int, page_no.number)

        result: list[Rect] = []
        blocks: Iterable[TextBlockPDF]
        if page_no is None:
            blocks = chain.from_iterable(
                [self._text_cache[i] for i in range(self.file.page_count)]
            )
        else:
            blocks = self._text_cache[page_no]
        for block in blocks:
            if (clip is None or clip.contains(block.rect)) and (
                min_len is None or len(block.text.strip()) >= min_len
            ):
                if copy_rects:
                    result.append(copy(block.rect))
                else:
                    result.append(block.rect)
        return result

    def get_drawing_rects(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:
        """
        Get drawing rectangles from a specific page.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return drawing rectangles for all document
        clip : Rect | None, optional
            Rectangle to clip the drawing blocks.
        copy_rects : bool, default=True
            Whether to return copies of the rectangles or references.

        Returns
        -------
        list[Rect]
            List of drawing rectangles.
        """

        return self._get_rects_by_type(
            rect_type="_drawings", page_no=page_no, clip=clip, copy_rects=copy_rects
        )

    def get_image_rects(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:
        """
        Get image rectangles from a specific page.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return image rectangles for all document
        clip : Rect | None, optional
            Rectangle to clip the image blocks.
        copy_rects : bool, default=True
            Whether to return copies of the rectangles or references.

        Returns
        -------
        list[Rect]
            List of image rectangles.
        """

        return self._get_rects_by_type(
            rect_type="_images", page_no=page_no, clip=clip, copy_rects=copy_rects
        )

    def get_figure_caption_rects(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:
        """
        Get figure caption rectangles from a specific page.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return figure caption rectangles for all document
        clip : Rect | None, optional
            Rectangle to clip the figure caption blocks.
        copy_rects : bool, default=True
            Whether to return copies of the rectangles or references.

        Returns
        -------
        list[Rect]
            List of figure caption rectangles.
        """

        return self._get_rects_by_type(
            rect_type="_figure_captions",
            page_no=page_no,
            clip=clip,
            copy_rects=copy_rects,
        )

    def get_paragraph_rects(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:
        """
        Get paragraph rectangles from a specific page.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return paragraph rectangles for all document
        clip : Rect | None, optional
            Rectangle to clip the paragraph blocks.
        copy_rects : bool, default=True
            Whether to return copies of the rectangles or references.

        Returns
        -------
        list[Rect]
            List of paragraph rectangles.
        """

        if isinstance(page_no, Page):
            page_no = cast(int, page_no.number)

        pages: Iterable[int]
        if page_no is None:
            pages = range(self.file.page_count)
        else:
            pages = [page_no]

        result: list[Rect] = []
        for page in pages:
            for rect in self._paragraphs_cache[page]:
                if clip is not None and not clip.contains(rect):
                    continue
                result.append(copy(rect) if copy_rects else rect)
        return result

    def _get_paragraphs_from_page(self, page_no: int) -> tuple[Rect, ...]:
        """
        Internal method to find the body paragraphs of a page.

        A paragraph is a text block of body width, grown over the lines that
        follow it: the last line of a paragraph is shorter than the rest and the
        continuation lines of a hanging indent are narrower than the body, so
        neither has the body width, yet both belong to the paragraph. Left out,
        they are seen as free-standing text and end up inside a figure that
        happens to sit under them.

        A line joins the paragraph above when it follows one `line_pitch` or one
        `paragraph_pitch` below it — the second for a body set with more leading
        than the rest of the document — lies within its horizontal span and is
        set in the font of the paragraph. The font is what keeps out the labels
        of a figure that happen to sit one pitch under the paragraph.

        A short document may have no body width to measure: a one-page
        supplementary file is a title, an author list and a stack of affiliations
        of a different width each, none of them repeated often enough for
        `_calc_paragraph_width` to cluster, which leaves the width at zero and
        every block outside it. Every prose block then seeds a paragraph
        instead — without it the page has no paragraph at all and the figure
        below them, bounded by the nearest paragraph above its caption, grows to
        the top of the page and swallows the whole title block.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[Rect, ...]
            Paragraph rectangles of the page.
        """

        blocks = self._text_cache[page_no]
        paragraphs = [block for block in blocks if self._is_paragraph_block(block)]
        rects = [copy(block.rect) for block in paragraphs]
        fonts = [self._block_font(block) for block in paragraphs]
        taken = {id(block) for block in paragraphs}
        pitches = {self.line_pitch, self.paragraph_pitch}

        extended = True
        while extended:
            extended = False
            for rect, font in zip(rects, fonts):
                for block in blocks:
                    if id(block) in taken or not block.lines:
                        continue
                    gap = block.lines[0].rect.y1 - rect.y1
                    if not any(
                        abs(gap - pitch)
                        <= max(1.0, self.LINE_PITCH_TOLERANCE * pitch)
                        for pitch in pitches
                    ):
                        continue
                    overlap = min(rect.x1, block.rect.x1) - max(rect.x0, block.rect.x0)
                    if overlap < self.MIN_PARAGRAPH_OVERLAP * block.rect.width:
                        continue
                    if not self._same_font(font, self._block_font(block)):
                        continue
                    rect.include_rect(block.rect)
                    taken.add(id(block))
                    extended = True
        return tuple(rects)

    def _is_paragraph_block(self, block: TextBlockPDF) -> bool:
        """
        Internal method to tell whether a block seeds a body paragraph.

        A block seeds one when it has the body width, or, in a document with no
        body width to measure, when it reads as prose.

        Parameters
        ----------
        block : TextBlockPDF
            Block to classify.

        Returns
        -------
        bool
            Whether the block seeds a paragraph.
        """

        if self.paragraph_width.mean > 0:
            return bool(
                self.paragraph_width.min
                <= block.rect.width
                <= self.paragraph_width.max
            )
        return self._is_prose(block)

    def get_figure_rects(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:

        return self._get_rects_by_type(
            rect_type="_figures", page_no=page_no, clip=clip, copy_rects=copy_rects
        )

    def get_figures(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_figures: bool = False,
    ) -> dict[str, FigurePDF]:
        """
        Get figures from the document.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return figures for all document
        clip : Rect | None, optional
            Rectangle to clip the figure blocks.
        copy_figures : bool, default=False
            Whether to return copies of the figures or references.

        Returns
        -------
        dict[str, FigurePDF]
            Dictionary of figures keyed by figure label.
        """

        if isinstance(page_no, Page):
            page_no = cast(int, page_no.number)

        pages: Iterable[int]
        if page_no is None:
            pages = range(self.file.page_count)
        else:
            pages = [page_no]

        result: dict[str, FigurePDF] = {}
        for page in pages:
            figures = self._figures_cache[page]
            for figure in figures:
                if clip is not None and not clip.contains(figure.rect):
                    continue
                fig_label = figure.caption.label
                # Check for duplicates
                if fig_label in result:
                    warn(
                        f"Duplicate figure label {fig_label} found on page {page} in {self}. Overwriting."
                    )

                self._figure_label_to_page_ind[fig_label] = page
                if copy_figures:
                    result[fig_label] = deepcopy(figure)
                else:
                    result[fig_label] = figure

        return result

    def get_table_rects(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:
        """
        Get table rectangles from a specific page.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return table rectangles for all document
        clip : Rect | None, optional
            Rectangle to clip the tables.
        copy_rects : bool, default=True
            Whether to return copies of the rectangles or references.

        Returns
        -------
        list[Rect]
            List of table rectangles.
        """

        return self._get_rects_by_type(
            rect_type="_tables", page_no=page_no, clip=clip, copy_rects=copy_rects
        )

    def get_table_caption_rects(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:
        """
        Get table caption rectangles from a specific page.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return them for all document
        clip : Rect | None, optional
            Rectangle to clip the table captions.
        copy_rects : bool, default=True
            Whether to return copies of the rectangles or references.

        Returns
        -------
        list[Rect]
            List of table caption rectangles.
        """

        return self._get_rects_by_type(
            rect_type="_table_captions",
            page_no=page_no,
            clip=clip,
            copy_rects=copy_rects,
        )

    def get_tables(
        self,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_tables: bool = False,
    ) -> dict[str, TablePDF]:
        """
        Get tables from the document.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return tables for all document
        clip : Rect | None, optional
            Rectangle to clip the tables.
        copy_tables : bool, default=False
            Whether to return copies of the tables or references.

        Returns
        -------
        dict[str, TablePDF]
            Dictionary of tables keyed by table label.
        """

        if isinstance(page_no, Page):
            page_no = cast(int, page_no.number)

        pages: Iterable[int]
        if page_no is None:
            pages = range(self.file.page_count)
        else:
            pages = [page_no]

        result: dict[str, TablePDF] = {}
        for page in pages:
            for table in self._tables_cache[page]:
                if clip is not None and not clip.contains(table.rect):
                    continue
                label = table.caption.label
                if label in result:
                    warn(
                        f"Duplicate table label {label} found on page {page} in {self}. Overwriting."
                    )
                self._table_label_to_page_ind[label] = page
                result[label] = deepcopy(table) if copy_tables else table
        return result

    def get_figure_drawings(
        self,
        figure_label: str | None = None,
        page_index: int | None = None,
        copy_objects: bool = False,
    ) -> list[DrawingObjectPDF]:
        """
        Get drawing objects for a specific figure.

        Parameters
        ----------
        figure_label : str | None, optional
            Label of the figure to get drawings for. If None, gets drawings for all
            figures.
        page_index : int | None, optional
            Page index to get figures from. If None, get all figures specified by figure_label.
        copy_objects : bool, default=False
            Whether to return copies of the drawing objects or references.

        Returns
        -------
        list[DrawingObjectPDF]
            List of drawing objects for the specified figure(s).
        """

        return cast(
            "list[DrawingObjectPDF]",
            self._get_figure_component_by_type(
                figure_label=figure_label,
                page_index=page_index,
                component_type="drawings",
                copy_objects=copy_objects,
            ),
        )

    def get_figure_images(
        self,
        figure_label: str | None = None,
        page_index: int | None = None,
        copy_objects: bool = False,
    ) -> list[ImageInfoPDF]:
        """
        Get image objects for a specific figure.

        Parameters
        ----------
        figure_label : str | None, optional
            Label of the figure to get images for. If None, gets images for all figures.
        page_index : int | None, optional
            Page index to get figures from. If None, get all figures specified by figure_label.
        copy_objects : bool, default=False
            Whether to return copies of the image objects or references.

        Returns
        -------
        list[ImageInfoPDF]
            List of image objects for the specified figure(s).
        """

        return cast(
            "list[ImageInfoPDF]",
            self._get_figure_component_by_type(
                figure_label=figure_label,
                page_index=page_index,
                component_type="images",
                copy_objects=copy_objects,
            ),
        )

    def get_figure_texts(
        self,
        figure_label: str | None = None,
        page_index: int | None = None,
        copy_objects: bool = False,
    ) -> list[TextBlockPDF]:
        """
        Get text, associated with a specific figure.

        Parameters
        ----------
        figure_label : str | None, optional
            Label of the figure to get text for. If None, gets text for all figures.
        page_index : int | None, optional
            Page index to get figures from. If None, get all figures specified by figure_label.
        copy_objects : bool, default=False
            Whether to return copies of the text blocks or references.

        Returns
        -------
        list[TextBlockPDF]
            List of text blocks for the specified figure(s).
        """

        return cast(
            "list[TextBlockPDF]",
            self._get_figure_component_by_type(
                figure_label=figure_label,
                page_index=page_index,
                component_type="text",
                copy_objects=copy_objects,
            ),
        )

    def _get_doi(self) -> str | None:
        """
        Get DOI of the article from the PDF metadata or article text.

        Sources are tried in order of decreasing reliability: the ``subject`` metadata
        key, the XMP metadata packet, then page text. Page text is scanned in page
        order and the first page yielding a DOI wins, because the article's own DOI
        appears in the front matter while the bibliography holds the DOIs of cited
        works. Scanning stops at the bibliography for the same reason.

        This method may fail if the doi is missing in metadata and text was OCRed poorly.
        This is especially true for scanned old documents.

        Returns
        -------
        str | None
            DOI string if found, otherwise None.
        """

        # Try to get DOI from PDF metadata
        metadata = cast(dict[str, str], self.file.metadata)
        if subj := metadata.get("subject"):
            doi = self.extract_doi_from_text(subj)
            if doi:
                return doi

        if xmp_doi := self._extract_doi_from_xmp():
            return xmp_doi

        for page_no in range(self.file.page_count):
            page_text = self._page(page_no).get_text()
            if not isinstance(page_text, str):
                continue
            # The bibliography lists the DOIs of cited works, never the article's own.
            refs_match = self.REFERENCES_HEADING_PATTERN.search(page_text)
            if refs_match is not None:
                page_text = page_text[: refs_match.start()]
            doi = self.extract_doi_from_text(page_text)
            if doi is not None:
                return doi
            if refs_match is not None:
                break
        warn(f"DOI not found in metadata or text of {self}.")
        return None

    def _extract_doi_from_xmp(self) -> str | None:
        """
        Get DOI from the XMP metadata packet of the PDF, if it carries one.

        Returns
        -------
        str | None
            DOI string if the packet exists and holds one, otherwise None.
        """

        try:
            xmp = self.file.get_xml_metadata()
        except Exception:  # noqa: BLE001 - pymupdf raises varied types without a packet
            return None
        if not xmp:
            return None
        match = self.XMP_DOI_PATTERN.search(xmp)
        return self._normalize_doi_candidate(match.group(1)) if match else None

    @classmethod
    def _normalize_doi_candidate(cls, candidate: str) -> str:
        """
        Strip the decorations publishers append to a DOI in running text.

        Parameters
        ----------
        candidate : str
            Raw DOI match.

        Returns
        -------
        str
            Lowercased DOI without a supplemental-material suffix or trailing
            sentence punctuation.
        """

        # ".../-/DCSupplemental" and friends address the supplement, not the article.
        return re.split(r"/-/", candidate.lower())[0].rstrip(".,;:)")

    @classmethod
    def extract_doi_candidates(cls, text: str) -> list[str]:
        """
        Find every DOI in a piece of text, most complete candidates first.

        A DOI broken across a line is rejoined before matching, and a candidate that
        is a strict prefix of another is dropped: a DOI wrapped after its ``.``
        separator would otherwise yield a valid-looking but truncated prefix.

        Parameters
        ----------
        text : str
            Text to search.

        Returns
        -------
        list[str]
            Deduplicated lowercase DOIs in order of appearance.
        """

        # Rejoin a line break inside a DOI. Anchoring on the characters a DOI cannot
        # end with keeps this from gluing ordinary prose together -- and a blanket
        # newline strip would let a match run on into the following word, since the
        # DOI character class contains letters and digits.
        text = re.sub(r"([./-])[ \t]*\n[ \t]*(?=[-._;()/:A-Za-z0-9])", r"\1", text)

        candidates: list[str] = []
        for match in cls.DOI_PATTERN.finditer(text):
            candidate = cls._normalize_doi_candidate(match.group())
            if candidate not in candidates:
                candidates.append(candidate)
        return [
            candidate
            for candidate in candidates
            if not any(
                other != candidate and other.startswith(candidate)
                for other in candidates
            )
        ]

    @classmethod
    def extract_doi_from_text(cls, text: str) -> str | None:
        """
        Find the first complete DOI in a piece of text.

        Parameters
        ----------
        text : str
            Text to search.

        Returns
        -------
        str | None
            DOI string if found, otherwise None.
        """

        candidates = cls.extract_doi_candidates(text)
        return candidates[0] if candidates else None

    def _figure_dpi(self, figure_label: str) -> int:
        """
        Resolution to rasterize a figure at, derived from the images it holds.

        The sharpest raster image in the figure sets the rate: rendering below
        its native resolution throws away detail that is present in the PDF,
        rendering above it only enlarges pixels. The result is clamped to
        `MIN_FIGURE_DPI`..`MAX_FIGURE_DPI`; a figure drawn entirely in vectors
        and text has no raster to measure and gets `VECTOR_FIGURE_DPI`.

        An image is measured over its whole placement box, not over the part
        its clipping path leaves visible — the pixels are spread over the box —
        so a cropped one reads at its true resolution.

        Parameters
        ----------
        figure_label : str
            Label of the figure.

        Returns
        -------
        int
            DPI to rasterize the figure at.
        """

        placements = [
            (image, Rect(0, 0, 1, 1) * image.transform)
            for image in self.get_figure_images(figure_label)
        ]
        dpis = [
            max(
                image.width * self.POINTS_PER_INCH / placement.width,
                image.height * self.POINTS_PER_INCH / placement.height,
            )
            for image, placement in placements
            if image.width
            and image.height
            and placement.width > 0
            and placement.height > 0
        ]
        if not dpis:
            return self.VECTOR_FIGURE_DPI
        return int(
            min(self.MAX_FIGURE_DPI, max(self.MIN_FIGURE_DPI, round(max(dpis))))
        )

    def extract_figure_drawings(
        self,
        figure_label: str | None = None,
        page_index: int | None = None,
        highlight_white: bool = True,
        output_path: PathLike | str = Path("figures"),
        dpi: int | Literal["auto"] = "auto",
    ) -> list[Path]:
        """
        Extract vector graphics component of a figure as a separate image.

        Parameters
        ----------
        figure_label : str | None, optional
            Label of the figure to extract. If None, extracts all figures.
        page_index : int | None, optional
            Page index to extract figures from. If None, extracts all figures
            specified by figure_label.
        highlight_white : bool, default=True
            Whether to highlight white drawings by placing a border around them.
        output_path : PathLike | str, default=Path("figures")
            Path to save the extracted figure images.
        dpi : int | "auto", default="auto"
            DPI for the output images. `"auto"` derives it per figure from the
            native resolution of the raster images inside it, see
            `_figure_dpi`.

        Returns
        -------
        list[Path]
            Paths to the saved figure images.
        """

        output_path = Path(output_path)

        figures = self.get_figures(page_index)
        if figure_label is not None:
            figures = {figure_label: figures[figure_label]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        paths: list[Path] = []
        output_path.mkdir(parents=True, exist_ok=True)
        for fig_label in figures:
            fig_dpi = self._figure_dpi(fig_label) if dpi == "auto" else dpi
            fig_drawings = self.get_figure_drawings(
                fig_label, copy_objects=highlight_white
            )
            if fig_drawings:
                if highlight_white:
                    for drawing in fig_drawings:
                        if drawing.fill == (1.0, 1.0, 1.0):
                            drawing.width = 1
                drawings_page = self._make_drawings(fig_drawings)
                pixmap = drawings_page.get_pixmap(
                    dpi=fig_dpi, clip=figures[fig_label].rect
                )

                path = output_path / Path(
                    f"{self.identifier}_fig_{fig_label}_vectors.png"
                )
                pixmap.save(path, output="PNG")
                paths.append(path)

        return paths

    def extract_figure_text(
        self,
        figure_label: str | None = None,
        page_index: int | None = None,
        output_path: PathLike | str = Path("figures"),
        dpi: int | Literal["auto"] = "auto",
    ) -> list[Path]:
        """
        Extract text component of a figure as a separate image.

        Parameters
        ----------
        figure_label : str | None, optional
            Label of the figure to extract. If None, extracts all figures.
        page_index : int | None, optional
            Page index to extract figures from. If None, extracts all figures
            specified by figure_label.
        output_path : PathLike | str, default=Path("figures")
            Path to save the extracted figure images.
        dpi : int | "auto", default="auto"
            DPI for the output images. `"auto"` derives it per figure from the
            native resolution of the raster images inside it, see
            `_figure_dpi`.

        Returns
        -------
        list[Path]
            Paths to the saved figure images.
        """

        output_path = Path(output_path)

        figures = self.get_figures(page_index)
        if figure_label is not None:
            figures = {figure_label: figures[figure_label]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        paths: list[Path] = []
        output_path.mkdir(parents=True, exist_ok=True)
        for fig_label in figures:
            fig_dpi = self._figure_dpi(fig_label) if dpi == "auto" else dpi
            figure_texts = self.get_figure_texts(fig_label)
            if figure_texts:
                figure_text_page = self._make_text(figure_texts)
                pixmap = figure_text_page.get_pixmap(
                    dpi=fig_dpi, clip=figures[fig_label].rect
                )

                path = output_path / Path(f"{self.identifier}_fig_{fig_label}_text.png")
                pixmap.save(path, output="PNG")
                paths.append(path)

        return paths

    def extract_figure_images(
        self,
        figure_label: str | None = None,
        page_index: int | None = None,
        output_path: PathLike | str = Path("figures"),
        dpi: int | Literal["auto"] = "auto",
    ) -> list[Path]:
        """
        Extract images component of a figure as a separate image.

        Parameters
        ----------
        figure_label : str | None, optional
            Label of the figure to extract. If None, extracts all figures.
        page_index : int | None, optional
            Page index to extract figures from. If None, extracts all figures
            specified by figure_label.
        output_path : PathLike | str, default=Path("figures")
            Path to save the extracted figure images.
        dpi : int | "auto", default="auto"
            DPI for the output images. `"auto"` derives it per figure from the
            native resolution of the raster images inside it, see
            `_figure_dpi`.

        Returns
        -------
        list[Path]
            Paths to the saved figure images.
        """

        output_path = Path(output_path)

        figures = self.get_figures(page_index)
        if figure_label is not None:
            figures = {figure_label: figures[figure_label]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        paths: list[Path] = []
        output_path.mkdir(parents=True, exist_ok=True)
        for fig_label in figures:
            fig_dpi = self._figure_dpi(fig_label) if dpi == "auto" else dpi
            figure_images = self.get_figure_images(fig_label)
            if figure_images:
                figure_image_page = self._make_image(figure_images)
                pixmap = figure_image_page.get_pixmap(
                    dpi=fig_dpi, clip=figures[fig_label].rect
                )

                path = output_path / Path(f"{self.identifier}_fig_{fig_label}_image.png")
                pixmap.save(path, output="PNG")
                paths.append(path)

        return paths

    def extract_figures(
        self,
        figure_label: str | None = None,
        page_index: int | None = None,
        output_path: PathLike | str = Path("figures"),
        extract_bases: bool = False,
        dpi: int | Literal["auto"] = "auto",
    ) -> dict[str, Path]:
        """
        Extract a specific figure as a separate images.

        Parameters
        ----------
        figure_label : str | None, optional
            Label of the figure to extract. If None, extracts all figures.
        page_index : int | None, optional
            Page index to extract figures from. If None, extracts all figures
            specified by figure_label.
        output_path : PathLike | str, default=Path("figures")
            Path to save the extracted figure images.
        extract_bases : bool, default=False
            Whether to also extract the base components (drawings, images, text)
        dpi : int | "auto", default="auto"
            DPI for the output images. `"auto"` derives it per figure from the
            native resolution of the raster images inside it, see
            `_figure_dpi`.

        Returns
        -------
        dict[str, Path]
            Paths to the saved figure images, keyed by figure label.
        """

        output_path = Path(output_path)

        figures = self.get_figures(page_index)
        if figure_label is not None:
            figures = {figure_label: figures[figure_label]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return {}

        paths: dict[str, Path] = {}
        output_path.mkdir(parents=True, exist_ok=True)
        for fig_label in figures:
            fig_dpi = self._figure_dpi(fig_label) if dpi == "auto" else dpi
            page_ind = self._figure_label_to_page_ind[fig_label]
            pixmap = self._page(page_ind).get_pixmap(
                dpi=fig_dpi, clip=figures[fig_label].rect
            )

            path = output_path / Path(f"{self.identifier}_fig_{fig_label}.png")
            pixmap.save(path, output="PNG")
            paths[fig_label] = path
            if extract_bases:
                self.extract_figure_drawings(
                    figure_label=fig_label,
                    page_index=page_ind,
                    output_path=output_path,
                    dpi=fig_dpi,
                )
                self.extract_figure_images(
                    figure_label=fig_label,
                    page_index=page_ind,
                    output_path=output_path,
                    dpi=fig_dpi,
                )
                self.extract_figure_text(
                    figure_label=fig_label,
                    page_index=page_ind,
                    output_path=output_path,
                    dpi=fig_dpi,
                )

        return paths

    def mark(
        self,
        elements: list[str] | str | None,
        page_number: int | None = None,
        output_path: PathLike | str | None = None,
        stroke_color: tuple[float, ...] = (1.0, 0.0, 0.0),
        stroke_width: float = 2.0,
    ) -> str:
        """
        Mark specified document elements on the PDF pages.

        Parameters
        ----------
        elements : list[str] | str | None
            Document elements to mark. Can be 'text', 'drawing', 'image',
            'figure_caption', 'header', or 'all'. If None, marks all elements.
        page_number : int | None, optional
            Page number (0-indexed) to mark. If None, marks all pages.
        output_path : PathLike | str | None, optional
            Path to save the marked PDF. If None, saves as 'marked_<original_filename>.pdf'.
        stroke_color : tuple[float, ...], default=(1.0, 0.0, 0.0)
            Color of the stroke used for marking.
        stroke_width : float, default=2.0
            Width of the stroke used for marking.

        Returns
        -------
        str
            Path to the saved marked PDF.

        """

        if elements is None:
            elements = "all"
        if not isinstance(elements, list):
            elements = [elements]

        if page_number is None:
            pages = [self._page(i) for i in range(self.file.page_count)]
        else:
            pages = [self._page(page_number)]

        for page in pages:
            rects = []
            for element in elements:
                if element not in DocumentElementsPDF:
                    warn(
                        f"Unknown document element: '{element}'."
                        + f" Allowed elements are: {[e.value for e in DocumentElementsPDF]}."
                    )
                    elements.remove(element)
                    continue
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.TEXT]:
                    rects.extend(self.get_text_rects(page.number, copy_rects=False))
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.DRAWING]:
                    rects.extend(self.get_drawing_rects(page.number, copy_rects=False))
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.IMAGE]:
                    rects.extend(self.get_image_rects(page.number, copy_rects=False))
                if element in [
                    DocumentElementsPDF.ALL,
                    DocumentElementsPDF.FIGURE_CAPTION,
                ]:
                    rects.extend(
                        self.get_figure_caption_rects(page.number, copy_rects=False)
                    )
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.PARAGRAPH]:
                    rects.extend(
                        self.get_paragraph_rects(page.number, copy_rects=False)
                    )
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.HEADER]:
                    rects.append(self.header_rect)
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.FOOTER]:
                    if not self.footer_rect.is_empty:
                        rects.append(self.footer_rect)
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.FIGURE]:
                    rects.extend(self.get_figure_rects(page.number, copy_rects=False))
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.TABLE]:
                    rects.extend(
                        self.get_table_rects(page.number, copy_rects=False)
                    )
                if element in [
                    DocumentElementsPDF.ALL,
                    DocumentElementsPDF.TABLE_CAPTION,
                ]:
                    rects.extend(
                        self.get_table_caption_rects(page.number, copy_rects=False)
                    )

            for rect in rects:
                shape = page.new_shape()
                shape.draw_rect(rect)
                shape.finish(color=stroke_color, width=stroke_width)
                shape.commit()

        if output_path is None:
            output_path = Path(f"marked_{self.identifier}").resolve()
        else:
            output_path = Path(output_path).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            file_id = self.identifier.split(".pdf")[0]
            output_path = output_path / Path(
                f"{file_id}_{'_'.join([element for element in elements])}.pdf"
            )

        self.file.save(str(output_path))
        return str(output_path)

    # Private methods

    def _get_rects_by_type(
        self,
        rect_type: str,
        page_no: Page | int | None = None,
        clip: Rect | None = None,
        copy_rects: bool = True,
    ) -> list[Rect]:

        req_blocks = getattr(self, rect_type + "_cache")

        if isinstance(page_no, Page):
            page_no = cast(int, page_no.number)

        result: list[Rect] = []
        if page_no is None:
            blocks = chain.from_iterable(
                [req_blocks[i] for i in range(self.file.page_count)]
            )
        else:
            blocks = req_blocks[page_no]
        for block in blocks:
            if clip is None or clip.contains(block.rect):
                if copy_rects:
                    result.append(copy(block.rect))
                else:
                    result.append(block.rect)
        return result

    def _get_figure_component_by_type(
        self,
        component_type: str,
        figure_label: str | None = None,
        page_index: int | None = None,
        copy_objects: bool = False,
    ) -> list[DrawingObjectPDF | ImageInfoPDF | TextBlockPDF]:
        """
        Internal method to get figure components (images, drawings, texts).
        """

        figures = self.get_figures(page_index)
        if figure_label is not None:
            figures = {figure_label: figures[figure_label]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        components: list[DrawingObjectPDF | ImageInfoPDF | TextBlockPDF] = []
        for fig_label in figures:
            page_ind = self._figure_label_to_page_ind[fig_label]
            page_components = getattr(self, f"_{component_type}_cache")[page_ind]

            components.extend(
                deepcopy(component) if copy_objects else component
                for component in page_components
                if figures[fig_label].rect.contains(component.rect)
            )
        return components

    def _get_raw_text_blocks_from_page(self, page_no: int) -> tuple[TextBlockPDF, ...]:
        """
        Internal method to get all non blank text blocks from a page.

        Line numbers are still present here. Use `_get_text_blocks_from_page`
        unless line numbers are what is being looked for.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[TextBlockPDF, ...]
            Non-blank text blocks on the page.
        """

        result: list[TextBlockPDF] = []
        text_blocks = [
            TextBlockPDF.from_dict(block)
            for block in self._page(page_no).get_text(option="dict")["blocks"]
            if block["type"] == 0
        ]
        for block in text_blocks:
            # Skip blank blocks
            if len(block.text) == 0 or block.text.isspace():
                continue
            result.append(block)
        return self._join_tile_fragments(tuple(result))

    def _join_tile_fragments(
        self, blocks: tuple[TextBlockPDF, ...]
    ) -> tuple[TextBlockPDF, ...]:
        """
        Internal method to put back together the lines of a page drawn in tiles.

        A PDF post-processor can redraw a page as a row of narrow vertical
        strips, each clipped to its own rectangle. Every line of text is then cut
        into a fragment per strip, and each strip becomes blocks of its own: a
        caption comes out as its first few words, is narrow enough to pass for a
        side caption, and no block is wide enough for a paragraph. A glyph
        crossing a seam is drawn once in each strip, so the fragment that
        continues a line starts on the same baseline with the glyph the line
        ends in — which no untiled page does. Such a page has its fragments
        joined into whole lines, the repeated glyph dropped, and its blocks put
        in reading order, as the order of the strips is not.

        Parameters
        ----------
        blocks : tuple[TextBlockPDF, ...]
            Non-blank text blocks of a page, in the order they were drawn.

        Returns
        -------
        tuple[TextBlockPDF, ...]
            The same blocks, or on a tiled page, blocks of whole lines.
        """

        # Fragments as (block number, line number, line), bucketed by baseline.
        by_baseline: dict[int, list[tuple[int, int, TextLinePDF]]] = {}
        for block_no, block in enumerate(blocks):
            for line_no, line in enumerate(block.lines):
                if line.wmode == 0 and line.dir[0] > 0.99 and line.spans:
                    by_baseline.setdefault(round(self._baseline(line)), []).append(
                        (block_no, line_no, line)
                    )

        def neighbours(line: TextLinePDF) -> Iterable[tuple[int, int, TextLinePDF]]:
            key = round(self._baseline(line))
            for near_key in (key - 1, key, key + 1):
                yield from by_baseline.get(near_key, ())

        repeated_glyphs = sum(
            1
            for fragments in by_baseline.values()
            for block_no, _, line in fragments
            if any(
                other_no != block_no and self._repeats_last_glyph(other, line)
                for other_no, _, other in neighbours(line)
            )
        )
        if repeated_glyphs < self.TILE_MIN_REPEATED_GLYPHS:
            return blocks

        # Grow each line from its leftmost fragment, strip by strip. Every line
        # is a candidate: a superscript cut by a seam goes on above the baseline.
        joined: dict[tuple[int, int], TextLinePDF] = {}
        consumed: set[tuple[int, int]] = set()
        fragments = sorted(
            chain.from_iterable(by_baseline.values()),
            key=lambda fragment: fragment[2].rect.x0,
        )
        for block_no, line_no, line in fragments:
            continued = [
                head for head in joined if self._is_tile_seam(joined[head], line)
            ]
            if continued:
                head = min(
                    continued, key=lambda head: abs(joined[head].rect.x1 - line.rect.x0)
                )
                joined[head] = self._join_fragments(joined[head], line)
                consumed.add((block_no, line_no))
            else:
                joined[(block_no, line_no)] = line

        result: list[TextBlockPDF] = []
        for block_no, block in enumerate(blocks):
            lines = [
                joined.get((block_no, line_no), line)
                for line_no, line in enumerate(block.lines)
                if (block_no, line_no) not in consumed
            ]
            if not lines:
                continue
            rect = Rect()
            for line in lines:
                rect.include_rect(line.rect)
            result.append(TextBlockPDF(rect=rect, lines=lines))
        result.sort(key=lambda block: (block.rect.y0, block.rect.x0))
        return tuple(result)

    @staticmethod
    def _baseline(line: TextLinePDF) -> float:
        """
        Internal method to get the baseline of a line.

        It is the baseline of the line's largest span: a line or a fragment of
        one can start with a superscript, or with a symbol set in an equation
        font on a baseline of its own.

        Parameters
        ----------
        line : TextLinePDF
            Line with at least one span.

        Returns
        -------
        float
            Vertical coordinate of the baseline.
        """

        return max(line.spans, key=lambda span: span.size).origin[1]

    @classmethod
    def _is_tile_seam(cls, line: TextLinePDF, fragment: TextLinePDF) -> bool:
        """
        Internal method to decide whether a fragment can continue a line.

        The fragment sits on the line's baseline, or goes on at the height the
        line breaks off at — a superscript can be cut by a seam — and starts where
        the line ends: at most a glyph before its end, where the strips overlap,
        and not past a word space after it.

        Parameters
        ----------
        line : TextLinePDF
            Line, possibly joined from several fragments already.
        fragment : TextLinePDF
            Fragment that may continue it.

        Returns
        -------
        bool
            Whether the fragment continues the line.
        """

        return (
            min(
                abs(cls._baseline(fragment) - cls._baseline(line)),
                abs(fragment.spans[0].origin[1] - line.spans[-1].origin[1]),
            )
            <= cls.TILE_SEAM_TOLERANCE
            and line.rect.x0 < fragment.rect.x0
            and line.rect.x1 - line.rect.height
            <= fragment.rect.x0
            <= line.rect.x1
            + max(cls.TILE_SEAM_TOLERANCE, cls.TILE_MAX_WORD_SPACE * line.rect.height)
        )

    @classmethod
    def _repeats_last_glyph(cls, line: TextLinePDF, fragment: TextLinePDF) -> bool:
        """
        Internal method to decide whether a fragment redraws a line's last glyph.

        Parameters
        ----------
        line : TextLinePDF
            Line that may end in the glyph.
        fragment : TextLinePDF
            Fragment that may start with it.

        Returns
        -------
        bool
            Whether the fragment continues the line starting with its last glyph.
        """

        last = line.spans[-1].text[-1:]
        return (
            cls._is_tile_seam(line, fragment)
            and fragment.rect.x0 < line.rect.x1 - cls.TILE_SEAM_TOLERANCE
            and last != ""
            and fragment.spans[0].text[:1] == last
        )

    @classmethod
    def _join_fragments(cls, line: TextLinePDF, fragment: TextLinePDF) -> TextLinePDF:
        """
        Internal method to append a fragment of a tiled page to its line.

        The glyph the two share is kept once, and the fragment's first span is
        merged into the line's last one when set in the same font: `text`
        separates spans with a space, which would split a word at the seam; a
        seam falling between two words in a gap no strip draws a space into
        gets one of its own.
        Whether a span is a superscript is not a font property but guessed by
        MuPDF from its neighbours, and the copies on either side of a seam can
        be guessed differently.

        Parameters
        ----------
        line : TextLinePDF
            Line the fragment continues.
        fragment : TextLinePDF
            Fragment to append.

        Returns
        -------
        TextLinePDF
            The joined line.
        """

        spans = list(fragment.spans)
        if cls._repeats_last_glyph(line, fragment):
            spans[0] = replace(spans[0], text=spans[0].text[1:])
            if not spans[0].text:
                spans.pop(0)
        joined_spans = list(line.spans)
        if spans:
            last, first = joined_spans[-1], spans[0]
            if (
                last.font == first.font
                and last.size == first.size
                and last.flags & ~pymupdf.TEXT_FONT_SUPERSCRIPT
                == first.flags & ~pymupdf.TEXT_FONT_SUPERSCRIPT
                and last.color == first.color
                and abs(last.origin[1] - first.origin[1]) <= cls.TILE_SEAM_TOLERANCE
            ):
                span_rect = Rect(last.rect)
                span_rect.include_rect(first.rect)
                gap = fragment.rect.x0 - line.rect.x1
                separator = (
                    " "
                    if gap > cls.TILE_MIN_WORD_SPACE * first.size
                    and not last.text[-1:].isspace()
                    and not first.text[:1].isspace()
                    else ""
                )
                joined_spans[-1] = replace(
                    last, rect=span_rect, text=last.text + separator + first.text
                )
                spans.pop(0)
        rect = Rect(line.rect)
        rect.include_rect(fragment.rect)
        return replace(line, rect=rect, spans=joined_spans + spans)

    def _get_text_blocks_from_page(self, page_no: int) -> tuple[TextBlockPDF, ...]:
        """
        Internal method to get all non blank text blocks from a page,
        without manuscript line numbers.

        A line number is a layout artifact rather than content: it corrupts the
        text of a block it was merged into, and its rectangle is a spurious
        neighbour for every heuristic that looks for the nearest element above
        or beside something. Blocks that consist only of line numbers are
        dropped; blocks that merely contain one keep their remaining lines and
        get a rectangle recomputed from them.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[TextBlockPDF, ...]
            Non-blank text blocks on the page, free of line numbers.
        """

        line_number_rects = self._line_number_rects[page_no]
        blocks = self._raw_text_cache[page_no]
        if not line_number_rects:
            return blocks

        result: list[TextBlockPDF] = []
        for block in blocks:
            kept = [
                line
                for line in block.lines
                if self._rect_key(line.rect) not in line_number_rects
            ]
            if len(kept) == len(block.lines):
                result.append(block)
                continue
            if not kept:
                continue
            rect = Rect()
            for line in kept:
                rect.include_rect(line.rect)
            result.append(TextBlockPDF(rect=rect, lines=kept))
        return tuple(result)

    @staticmethod
    def _rect_key(rect: Rect) -> tuple[float, float, float, float]:
        """
        Internal method to make a hashable key out of a rectangle.

        Parameters
        ----------
        rect : Rect
            Rectangle to convert.

        Returns
        -------
        tuple[float, float, float, float]
            Corner coordinates of the rectangle.
        """

        return (rect.x0, rect.y0, rect.x1, rect.y1)

    def _find_line_numbers(
        self,
    ) -> dict[int, frozenset[tuple[float, float, float, float]]]:
        """
        Internal method to find manuscript line numbers in the document.

        A line number belongs to a column of bare integers, narrow, aligned on
        one of its edges, whose values grow down each page. Numbering marks
        several lines of a page, which is what separates it from a page number
        in the footer; and it never restarts within a page, which is what
        separates it from a numbered list.

        Returns
        -------
        dict[int, frozenset[tuple[float, float, float, float]]]
            Coordinates of the line number rectangles, keyed by page number.
        """

        page_count = self.file.page_count
        candidates: list[tuple[int, int, TextLinePDF]] = []
        for page_no in range(page_count):
            page_width = cast(Rect, self._page(page_no).rect).width
            max_width = page_width * self.LINE_NUMBER_MAX_WIDTH
            for block in self._raw_text_cache[page_no]:
                for line in block.lines:
                    text = line.text.strip()
                    if (
                        self.LINE_NUMBER_PATTERN.fullmatch(text)
                        and line.rect.width < max_width
                    ):
                        candidates.append((page_no, int(text), line))

        found: dict[int, set[tuple[float, float, float, float]]] = {
            page_no: set() for page_no in range(page_count)
        }
        # Line numbers can be aligned on either edge: a column of right-aligned
        # numbers shares x1, a column of left-aligned ones shares x0.
        for align_right in (False, True):
            for column in self._group_aligned_lines(candidates, align_right):
                if not self._is_line_number_column(column, page_count):
                    continue
                for page_no, _, line in column:
                    found[page_no].add(self._rect_key(line.rect))
        return {page_no: frozenset(rects) for page_no, rects in found.items()}

    def _line_number_gutter(self, page_no: int) -> tuple[float, float]:
        """
        Internal method to get the horizontal search bounds of a page.

        A line number column occupies a strip of the margin that no figure can
        reach — printed content there would collide with the numbers. Excluding
        that strip keeps it out of a figure rectangle that could not be narrowed
        down to its contents, and so out of the image `extract_figures` renders
        from that rectangle.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[float, float]
            Left and right bounds of the searchable area of the page.
        """

        page_rect = cast(Rect, self._page(page_no).rect)
        left_bound = page_rect.x0
        right_bound = page_rect.x1

        line_number_rects = self._line_number_rects[page_no]
        if not line_number_rects:
            return left_bound, right_bound

        middle = (page_rect.x0 + page_rect.x1) / 2
        left_column = [rect for rect in line_number_rects if rect[2] <= middle]
        right_column = [rect for rect in line_number_rects if rect[0] > middle]
        if left_column:
            left_bound = max(rect[2] for rect in left_column) + self.MARGIN
        if right_column:
            right_bound = min(rect[0] for rect in right_column) - self.MARGIN
        return left_bound, right_bound

    @classmethod
    def _group_aligned_lines(
        cls,
        candidates: list[tuple[int, int, TextLinePDF]],
        align_right: bool,
    ) -> list[list[tuple[int, int, TextLinePDF]]]:
        """
        Internal method to group candidate lines sharing an aligned edge.

        Parameters
        ----------
        candidates : list[tuple[int, int, TextLinePDF]]
            Candidate lines as (page number, value, line).
        align_right : bool
            Whether to group by the right edge of the line instead of the left.

        Returns
        -------
        list[list[tuple[int, int, TextLinePDF]]]
            Candidates grouped into columns.
        """

        if not candidates:
            return []

        def edge(candidate: tuple[int, int, TextLinePDF]) -> float:
            rect = candidate[2].rect
            return rect.x1 if align_right else rect.x0

        ordered = sorted(candidates, key=edge)
        columns: list[list[tuple[int, int, TextLinePDF]]] = []
        column = [ordered[0]]
        for candidate in ordered[1:]:
            if edge(candidate) - edge(column[-1]) <= cls.LINE_NUMBER_ALIGN_TOLERANCE:
                column.append(candidate)
            else:
                columns.append(column)
                column = [candidate]
        columns.append(column)
        return columns

    @classmethod
    def _is_line_number_column(
        cls,
        column: list[tuple[int, int, TextLinePDF]],
        page_count: int,
    ) -> bool:
        """
        Internal method to decide whether a column of numbers is line numbering.

        Parameters
        ----------
        column : list[tuple[int, int, TextLinePDF]]
            Candidates of one column, as (page number, value, line).
        page_count : int
            Number of pages in the document.

        Returns
        -------
        bool
            Whether the column is a line number column.
        """

        if len(column) < cls.LINE_NUMBER_MIN_COUNT:
            return False

        numbers_per_page: dict[int, int] = {}
        for page_no in {page for page, _, _ in column}:
            values = [
                value
                for page, value, _ in sorted(column, key=lambda item: item[2].rect.y0)
                if page == page_no
            ]
            # A numbered list restarts its count, line numbering never does.
            if any(
                later <= earlier for earlier, later in zip(values, values[1:])
            ):
                return False
            numbers_per_page[page_no] = len(values)

        multiline_pages = sum(1 for count in numbers_per_page.values() if count >= 2)
        return multiline_pages >= max(
            2, page_count * cls.LINE_NUMBER_MULTILINE_PAGE_FRACTION
        )

    def _get_drawing_objs_from_page(self, page_no: int) -> tuple[DrawingObjectPDF, ...]:
        """
        Internal method to get all drawing objects from a page.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[DrawingObjectPDF, ...]
            Drawing objects on the page.
        """

        page = self._page(page_no)
        drawings = tuple(
            [
                DrawingObjectPDF(**{k: v for k, v in dr.items() if v is not None})
                for dr in page.get_drawings()
            ]
        )
        return drawings

    def _get_image_info_from_page(self, page_no: int) -> tuple[ImageInfoPDF, ...]:
        """
        Internal method to get all image info from a page.

        An image is measured by the part of it left visible by the clipping path
        it is drawn through, not by its whole placement box: a word processor
        can place a panel letter as a page-sized picture clipped down to the
        letter, and its placement box then reaches past the caption and keeps
        the letter out of every figure.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[ImageInfoPDF, ...]
            Image info for images on the page, excluding oversized ones (see
            `MAX_IMAGE_AREA`).
        """

        page = self._page(page_no)
        page_area = abs(page.rect)
        # What `Page.get_image_info(hashes=True, xrefs=True)` does, with clipping.
        textpage = page.get_textpage(
            flags=pymupdf.TEXT_PRESERVE_IMAGES | pymupdf.TEXT_CLIP
        )
        infos = cast(list[dict], textpage.extractIMGINFO(hashes=True))
        xrefs = {
            Pixmap(self.file, item[0]).digest: item[0]
            for item in self.file.get_page_images(page_no)
        }
        for info in infos:
            info["xref"] = xrefs.get(info["digest"], 0)
            # A clip trimming no more than a margin off a side of the image — a
            # frame, the page edge — says nothing about where it is; that side
            # keeps its placement.
            placement = Rect(0, 0, 1, 1) * pymupdf.Matrix(info["transform"])
            info["bbox"] = tuple(
                shown if abs(shown - placed) > self.MARGIN else placed
                for placed, shown in zip(tuple(placement), info["bbox"])
            )
        all_images = tuple([ImageInfoPDF.from_dict(info) for info in infos])
        page_size_images = [
            image
            for image in all_images
            if abs(image.rect) > page_area * self.MAX_IMAGE_AREA
        ]
        if page_size_images:
            warn("Too large images detected. Figure extraction unreliable")
        return tuple(
            image
            for image in all_images
            if abs(image.rect) < page_area * self.MAX_IMAGE_AREA
        )

    def _make_drawings(
        self, drawings: Iterable[DrawingObjectPDF], page: Page | None = None
    ) -> Page:
        """
        Internal method to render drawing objects.

        Parameters
        ----------
        drawings : Iterable[DrawingObjectPDF]
            Drawing objects to render.
        page : Page | None
            Page to render drawings on. If None, creates a new blank page.

        Raises
        ------
        ValueError
            If a drawing contains an unknown drawing item type.

        Returns
        -------
        Page
            Page with rendered drawings.
        """

        if page is None:
            outpdf = pymupdf.open()
            page = outpdf.new_page(
                width=self._page(0).rect.width, height=self._page(0).rect.height
            )
        shape = page.new_shape()
        for drawing in drawings:
            for item in drawing.items:
                if item[0] == "l":  # line
                    shape.draw_line(item[1], item[2])
                elif item[0] == "re":  # rectangle
                    shape.draw_rect(item[1])
                elif item[0] == "c":  # curve
                    shape.draw_bezier(item[1], item[2], item[3], item[4])
                elif item[0] == "qu":  # quad
                    shape.draw_quad(item[1])
                else:
                    raise ValueError(f"Unknown drawing item type: {item[0]}")
            shape.finish(**drawing.finishing_opts())
        shape.commit()
        return page

    def _make_text(
        self, text_blocks: Iterable[TextBlockPDF], page: Page | None = None
    ) -> Page:
        """
        Internal method to render text objects.

        Parameters
        ----------
        text_blocks : Iterable[TextBlockPDF]
            Text blocks to render.
        page : Page | None
            Page to render text on. If None, creates a new blank page.

        Returns
        -------
        Page
            Page with rendered text.
        """

        if page is None:
            outpdf = pymupdf.open()
            page = outpdf.new_page(
                width=self._page(0).rect.width, height=self._page(0).rect.height
            )
        shape = page.new_shape()
        for block in text_blocks:
            for text_span in block.insert_text_dicts():
                shape.insert_text(**text_span)
        shape.commit()
        return page

    def _make_image(
        self, images: Iterable[ImageInfoPDF], page: Page | None = None
    ) -> Page:
        """
        Internal method to render image objects.

        Parameters
        ----------
        images : Iterable[ImageInfoPDF]
            Image objects to render.
        page : Page | None
            Page to render images on. If None, creates a new blank page.

        Returns
        -------
        Page
            Page with rendered images.
        """

        if page is None:
            outpdf = pymupdf.open()
            page = outpdf.new_page(
                width=self._page(0).rect.width, height=self._page(0).rect.height
            )
        for image in images:
            image_pm = Pixmap(self.file, image.xref)
            page.insert_image(image.rect, pixmap=image_pm)
        return page

    def _calc_paragraph_width(self, eps: float = 4.0) -> Size:
        """
        Internal method to calculate paragraph width. Mutates self.

        To calculate width all text paragraphs are clustered based on their width.
        DBSCAN is used for clustering.

        Only running text takes part. The rows of a table cluster at a width of
        their own, and a tall table can carry more total height than the body
        text it is printed among, which would leave no block on any page matching
        the "paragraph" width. Selecting on prose keeps them out without needing
        the tables to have been found first — which would be circular, since
        finding them uses this width.

        Only horizontal text takes part — within a few degrees, as a scanned
        page is often slightly skewed. A publisher's download stamp printed
        up the page margin ("Downloaded from pubs.acs.org/...") is a block of
        the same few points' width on every page; on a short document it
        outweighs the body text and becomes the "paragraph" width.

        Parameters
        ----------
        eps : float
            The maximum distance between two samples for one to be considered as
            in the neighborhood of the other.

        Returns
        -------
        Size
            Mean, min, max width of the paragraph-width cluster. All zero when
            the document has no running text, or too little of it for any width
            to repeat — `_get_paragraphs_from_page` then falls back to prose.
        """

        rects: list[Rect] = [
            block.rect
            for page_no in range(self.file.page_count)
            for block in self._text_cache[page_no]
            if self._is_prose(block)
            and all(line.dir[0] > 0.99 for line in block.lines)
        ]
        if not rects:
            logger.debug(
                f"No running text found in {self}."
                + " Paragraphs fall back to prose, of which there is none."
            )
            return Size(mean=0.0, min=0.0, max=0.0)

        text_blocks = pd.DataFrame(
            [[rect.x0, rect.y0, rect.x1, rect.y1] for rect in rects],
            columns=["x0", "y0", "x1", "y1"],
        )
        text_blocks["width"] = text_blocks["x1"] - text_blocks["x0"]
        text_blocks["height"] = text_blocks["y1"] - text_blocks["y0"]

        clustering = DBSCAN(eps=eps)
        text_blocks["groups"] = clustering.fit_predict(text_blocks[["width"]])

        valid_blocks = text_blocks[text_blocks["groups"] != -1]
        if valid_blocks.empty:
            logger.debug(
                f"No block width repeats often enough to cluster in {self}."
                + " Paragraphs fall back to prose."
            )
            return Size(mean=0.0, min=0.0, max=0.0)

        clusters = valid_blocks.pivot_table(
            index="groups",
            aggfunc={"width": ["mean", "min", "max", "count"], "height": "sum"},  # type: ignore
        ).sort_values(by=("height", "sum"), ascending=False)

        largest_cluster = clusters.iloc[0]
        width_mean = largest_cluster[("width", "mean")]
        width_min = largest_cluster[("width", "min")]
        width_max = largest_cluster[("width", "max")]
        return Size(mean=width_mean, min=width_min, max=width_max)

    @staticmethod
    def _caption_label(match: re.Match[str]) -> str:
        """
        Build a figure label from a `CAPTION_PATTERN` match.

        The label is the figure number with leading zeros dropped, prefixed with
        `S` for a supplementary figure. Both spellings of "supplementary" mark
        the same thing, so `Supplementary Fig. S1` is labelled `S1`, not `SS1`,
        and `Figure 1S` — the `S` set after the number — is labelled `S1` too.

        Parameters
        ----------
        match : re.Match[str]
            A match of `CAPTION_PATTERN`.

        Returns
        -------
        str
            Figure label, e.g. `1` or `S1`.
        """

        supplementary = (
            match.group("supp") or match.group("prefix") or match.group("suffix")
        )
        prefix = "S" if supplementary else ""
        return f"{prefix}{int(match.group('number'))}"

    @staticmethod
    def _normalize_caption_whitespace(text: str) -> str:
        """
        Collapse whitespace runs and fix spacing around punctuation.

        Depending on the installed pymupdf version, a text line can be split
        into many single-word spans plus dedicated whitespace-only spans
        instead of fewer, coarser spans with whitespace embedded in their
        text. A naive per-span join then inserts extra spaces at those
        boundaries, producing double/triple spaces and stray spaces before
        punctuation or after an opening parenthesis. This is used only when
        assembling figure caption text.

        Parameters
        ----------
        text : str
            Text to normalize.

        Returns
        -------
        str
            Normalized text.
        """

        # Collapse any run of whitespace (space, tab, thin space, nbsp, ...) to one space.
        text = re.sub(r"\s+", " ", text)
        # Drop the space before a closing punctuation mark, e.g. "word ." -> "word.".
        text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
        # Drop the space right after an opening parenthesis, e.g. "( word" -> "(word".
        text = re.sub(r"([(])\s+", r"\1", text)
        return text.strip()

    @classmethod
    def _caption_line_text(cls, line: TextLinePDF) -> str:
        """
        Assemble a single line's text from its spans for figure captions.

        A soft hyphen (U+00AD) marks an optional break point and is never
        real content. One in the middle of the line never took effect and
        is dropped outright; one at the very end reflects an actual
        line-wrap and is kept as a single trailing marker for
        `_join_caption_lines` to resolve, the same way a plain "-" is.

        Parameters
        ----------
        line : TextLinePDF
            Line to assemble text from.

        Returns
        -------
        str
            Normalized line text (see `_normalize_caption_whitespace`).
        """

        raw = " ".join(span.text for span in line.spans)
        text = cls._normalize_caption_whitespace(raw)
        ends_with_soft_hyphen = text.endswith("\xad")
        text = text.replace("\xad", "").rstrip()
        if ends_with_soft_hyphen:
            text += "\xad"
        return text

    @cached_property
    def _compound_hyphen_pairs(self) -> frozenset[tuple[str, str]]:
        """
        Word pairs confirmed to be genuine hyphenated compounds.

        Built by scanning every line of the document for hyphen-joined word
        chains (e.g. "Si-NPs") that occur away from the end of a physical
        PDF line. If a pair only ever appears where its hyphen coincides
        with a line end, it is more likely a line-wrap artifact than a real
        compound, so it is excluded here; `_join_caption_lines` then merges
        it (dropping the hyphen) instead of keeping it when assembling
        figure captions.

        Returns
        -------
        frozenset[tuple[str, str]]
            Lowercased (word1, word2) pairs confirmed to appear hyphenated
            somewhere other than a line-wrap point.
        """

        # Matches a hyphen-joined chain of word characters, e.g. "Si-NPs" or
        # "DARPin_9-29-tagRFP" (each dash-separated segment on its own).
        token_pattern = re.compile(r"[^\W_]+(?:-[^\W_]+)*")
        pairs: set[tuple[str, str]] = set()
        for page_no in range(self.file.page_count):
            for block in self._text_cache[page_no]:
                for line in block.lines:
                    text = self._caption_line_text(line)
                    stripped = text.rstrip()
                    for tok_match in token_pattern.finditer(text):
                        token = tok_match.group(0)
                        if "-" not in token:
                            continue
                        is_line_final_token = (
                            tok_match.end() == len(stripped) and stripped.endswith("-")
                        )
                        parts = token.split("-")
                        for i in range(len(parts) - 1):
                            is_last_pair = i == len(parts) - 2
                            if is_last_pair and is_line_final_token:
                                continue
                            pairs.add((parts[i].lower(), parts[i + 1].lower()))
        return frozenset(pairs)

    def _join_caption_lines(self, lines: Iterable[TextLinePDF]) -> str:
        """
        Join a figure caption's lines into clean text.

        Mirrors the usual per-line join, but additionally collapses
        whitespace artifacts introduced by span-splitting (see
        `_caption_line_text`) and resolves hyphens at line-wrap boundaries.
        A soft hyphen (U+00AD) is always dropped, since it never carries
        content. A plain "-" is resolved using `_compound_hyphen_pairs`:
        dropped when it looks like a wrapped word (e.g. "nanopar-" /
        "ticles" -> "nanoparticles") and kept when the pair is a confirmed
        compound elsewhere in the document (e.g. "Si-" / "NPs" -> "Si-NPs").

        Parameters
        ----------
        lines : Iterable[TextLinePDF]
            Lines to join, in reading order.

        Returns
        -------
        str
            Assembled caption text.
        """

        # Matches the last run of word characters in a string, e.g. the "NPs"
        # in "...Si-NPs" -- used to get the word right before a line-final hyphen.
        word_end_pattern = re.compile(r"[^\W_]+$")
        # Matches the first run of word characters in a string, e.g. the
        # "ticles" in "ticles from..." -- the word right after the line break.
        word_start_pattern = re.compile(r"[^\W_]+")
        compound_pairs = self._compound_hyphen_pairs

        result = ""
        for line in lines:
            text = self._caption_line_text(line)
            if not text:
                continue
            if not result:
                result = text
                continue
            if result.endswith("\xad"):
                result = result[:-1] + text
            elif result.endswith("-"):
                w1_match = word_end_pattern.search(result[:-1])
                w2_match = word_start_pattern.match(text)
                keep_hyphen = (
                    w1_match is not None
                    and w2_match is not None
                    and (w1_match.group(0).lower(), w2_match.group(0).lower())
                    in compound_pairs
                )
                if not keep_hyphen:
                    result = result[:-1]
                result += text
            else:
                result += " " + text
        return result

    def _find_figure_captions(self) -> dict[int, tuple[FigureCaptionPDF, ...]]:
        """Internal method to find figure captions on page."""

        return self._find_captions(self.CAPTION_PATTERN)

    def _find_captions(
        self, pattern: re.Pattern[str]
    ) -> dict[int, tuple[FigureCaptionPDF, ...]]:
        """
        Internal method to find the captions matching a pattern.

        Figure and table captions are laid out the same way — a label, then the
        caption text, possibly split over several blocks — and are found the same
        way, by the pattern that opens them.

        Parameters
        ----------
        pattern : re.Pattern[str]
            Pattern of the caption opening, with the groups of `CAPTION_PATTERN`.

        Returns
        -------
        dict[int, tuple[FigureCaptionPDF, ...]]
            Captions keyed by page number.
        """

        def _find_figure_captions_in_page(
            page: Page,
        ) -> dict[int, tuple[FigureCaptionPDF, ...]]:
            vertical_thr = 1
            fig_captures = []
            text_blocks = self._text_cache[page.number]
            num_blocks = len(text_blocks)
            i = 0
            while i < num_blocks:
                block = text_blocks[i]
                text = block.text.strip()
                match = pattern.match(text)
                if match:
                    # Some paragraphs can also begin with `pattern`, but in almost all cases
                    # font of `pattern` in figure caption is different from font of paragraph.
                    # Therefore we save font properties for the mathed block for later use.
                    font_props = (
                        block.lines[0].spans[0].char_flags,
                        block.lines[0].spans[0].flags,
                        block.lines[0].spans[0].font,
                    )

                    # I have encountered three possible cases:
                    # 1. Block contains full figure caption. This is the easiest case.
                    # We just take its text and bounding box.
                    # 2. Block contains only the pattern (e.g., "Fig. 1"). In this case,
                    # we need to look for subsequent blocks to complete the caption.
                    # 3. Block contains some lines of the capture, but not all of them.
                    # Cases 2 and 3 can occur simultaneously.

                    # If only pattern was found, include the next block
                    pattern_text = match.group(0)
                    if text == pattern_text:
                        i += 1
                        if i == num_blocks:
                            warn("Figure caption not fully recognized")
                            break
                        block = block + text_blocks[i]

                    # Case 3: caption split in several blocks. Check if the first line
                    # of the next block is one line down from the last line of current block.
                    # Once two lines of the caption are known, their distance is
                    # its leading, and a line set at any other is not part of
                    # it: a single-spaced caption is followed by the paragraph
                    # of a double-spaced body one `line_pitch` below.
                    caption_pitch: float | None = None
                    if len(block.lines) > 1:
                        caption_pitch = (
                            block.lines[-1].rect.y1 - block.lines[-2].rect.y1
                        )
                    # The next line is measured from the last full line of the
                    # caption, not from the last line joined: that can be the
                    # tail of a line, raised above it by a superscript.
                    last_rect = block.lines[-1].rect
                    i += 1
                    while i < num_blocks:
                        next_block = text_blocks[i]
                        # Blocks can include inner empty lines, so we should add lines by one.
                        capture_extended = False
                        lines_consumed = 0
                        for j in range(len(next_block.lines)):
                            next_line = next_block.lines[j]
                            # The tail of the last line, split off into a block
                            # of its own: a symbol or equation font breaks a line
                            # at a Greek letter or a superscript (the β of β-CD,
                            # the 1 of E¹₂g), possibly more than once. It sits on
                            # the same line and starts where the line so far
                            # ends — the line of the other column starts past a
                            # gutter. It is merged into that line rather than
                            # added as a line of its own, which would break the
                            # caption's text and its line count at the tail.
                            line_so_far = block.lines[-1]
                            same_line = min(
                                next_line.rect.y1, last_rect.y1
                            ) - max(next_line.rect.y0, last_rect.y0) > 0.5 * min(
                                next_line.rect.height, last_rect.height
                            )
                            if (
                                same_line
                                and next_line.rect.x0
                                <= line_so_far.rect.x1 + vertical_thr
                                and next_line.rect.x1 > line_so_far.rect.x1
                            ):
                                line_rect = Rect(line_so_far.rect)
                                line_rect.include_rect(next_line.rect)
                                block_rect = Rect(block.rect)
                                block_rect.include_rect(next_line.rect)
                                block = TextBlockPDF(
                                    rect=block_rect,
                                    lines=block.lines[:-1]
                                    + [
                                        replace(
                                            line_so_far,
                                            rect=line_rect,
                                            spans=line_so_far.spans
                                            + next_line.spans,
                                        )
                                    ],
                                )
                                capture_extended = True
                                lines_consumed += 1
                                continue
                            gap = next_line.rect.y1 - last_rect.y1
                            # Beneath the caption, and one pitch below or
                            # single-spaced: a caption is often set single-spaced
                            # under a body set at one and a half or double
                            # spacing, its lines anywhere from one line height
                            # to `MAX_SINGLE_SPACING` line heights apart.
                            # Once the caption's own spacing is known, a line set
                            # wider is not part of it, but one set single-spaced
                            # still is: a word processor can close up the last
                            # line of a paragraph set at one and a half spacing.
                            height = last_rect.height
                            single_spaced = (
                                (1 - self.LINE_PITCH_TOLERANCE) * height
                                <= gap
                                <= self.MAX_SINGLE_SPACING * height
                            )
                            if caption_pitch is not None:
                                consecutive = (
                                    abs(gap - caption_pitch)
                                    < max(
                                        vertical_thr,
                                        self.LINE_PITCH_TOLERANCE * caption_pitch,
                                    )
                                    or single_spaced
                                )
                            else:
                                consecutive = (
                                    abs(gap - self.line_pitch)
                                    < max(
                                        vertical_thr,
                                        self.LINE_PITCH_TOLERANCE * self.line_pitch,
                                    )
                                    or single_spaced
                                )
                            if consecutive and min(
                                next_line.rect.x1, last_rect.x1
                            ) > max(next_line.rect.x0, last_rect.x0):
                                caption_pitch = gap
                                block = block + next_line
                                last_rect = next_line.rect
                                capture_extended = True
                                lines_consumed += 1
                            else:
                                break
                        # Full block was consumed, check the next block
                        if lines_consumed == len(next_block.lines):
                            i += 1
                            continue
                        # If we did not extend caption, do not consume next block
                        if not capture_extended:
                            i -= 1
                        break
                    # Assemble clean caption text (whitespace-normalized,
                    # line-wrap hyphens resolved) and remove the matched pattern.
                    clean_text = self._join_caption_lines(block.lines)
                    clean_match = pattern.match(clean_text)
                    text = (
                        clean_text[clean_match.end() :].strip()
                        if clean_match is not None
                        else clean_text.replace(pattern_text, "").strip()
                    )
                    fig_captures.append(
                        FigureCaptionPDF(
                            matched_pattern=pattern_text,
                            label=self._caption_label(match),
                            font_props=font_props,
                            text=text,
                            lines_no=len(block.lines),
                            rect=copy(block.rect),
                        )
                    )
                i += 1
            return {page.number: tuple(fig_captures)}

        captions: dict[int, tuple[FigureCaptionPDF, ...]] = {}
        for page_no in range(self.file.page_count):
            captions.update(_find_figure_captions_in_page(self._page(page_no)))
        # Some text paragraphs can start the same way as figure caption.
        # Text paragraph usually have their font_flag different from font_flag
        # of matched_pattern in real figure caption.
        # Therefore we only use found FigureCaption with the most common font_flag
        most_common_flags = Counter(
            caption.font_props
            for page_captions in captions.values()
            for caption in page_captions
        ).most_common(1)
        if not most_common_flags:
            return captions
        most_common_flag = most_common_flags[0][0]
        result = {
            page_no: tuple(
                [
                    caption
                    for caption in page_captions
                    if caption.font_props == most_common_flag
                ]
            )
            for page_no, page_captions in captions.items()
        }
        return result

    @staticmethod
    def _font_family(font: str) -> str:
        """
        Internal method to reduce a font name to the family it belongs to.

        Parameters
        ----------
        font : str
            Font name as the PDF gives it, possibly with a subset prefix
            (`ABCDEF+Arial-BoldMT`).

        Returns
        -------
        str
            Lowercased family name, e.g. `arial` or `timesnewroman`.
        """

        name = font.rsplit("+", 1)[-1].lower()
        return ArticlePDF.FONT_STYLE_PATTERN.sub("", name, count=1)

    @staticmethod
    def _block_font(block: TextBlockPDF) -> tuple[str, float] | None:
        """
        Internal method to find the font most of a block is set in.

        Parameters
        ----------
        block : TextBlockPDF
            Block to measure.

        Returns
        -------
        tuple[str, float] | None
            Font family, as `_font_family` gives it, and size, or None for a
            block with no visible text.
        """

        chars: Counter[tuple[str, float]] = Counter()
        for line in block.lines:
            for span in line.spans:
                font = (ArticlePDF._font_family(span.font), round(span.size, 1))
                chars[font] += len(span.text.strip())
        most_common = chars.most_common(1)
        if not most_common or most_common[0][1] == 0:
            return None
        return most_common[0][0]

    @staticmethod
    def _same_font(
        font: tuple[str, float] | None, other: tuple[str, float] | None
    ) -> bool:
        """
        Internal method to tell whether two fonts are the same, in any cut.

        Parameters
        ----------
        font, other : tuple[str, float] | None
            Font family and size, as `_block_font` gives them.

        Returns
        -------
        bool
            Whether both are known, of one family and of one size within
            `FONT_SIZE_TOLERANCE`.
        """

        if font is None or other is None:
            return False
        return (
            font[0] == other[0]
            and abs(font[1] - other[1]) <= ArticlePDF.FONT_SIZE_TOLERANCE
        )

    def _is_body_text(self, block: TextBlockPDF) -> bool:
        """
        Internal method to tell whether a block is set in the body font.

        A block qualifies if most of it is set in the body family, in any cut,
        and no smaller than the body size: a heading is set larger or bold, and
        still reads as document text. The block is judged by the font most of it
        is set in, not by every span — the subscript of a chemical formula
        (`WS₂`) is set smaller, and a title merged into the block of the heading
        beneath it may be set in a typeface of its own, and neither makes the
        heading any less a heading.

        Parameters
        ----------
        block : TextBlockPDF
            Block to classify.

        Returns
        -------
        bool
            Whether the block is set in the body font.
        """

        family, size = self.body_font
        block_font = self._block_font(block)
        return (
            block_font is not None
            and block_font[0] == family
            and block_font[1] >= size - self.FONT_SIZE_TOLERANCE
        )

    @staticmethod
    def _is_prose(block: TextBlockPDF) -> bool:
        """
        Internal method to tell running text from the cells of a table.

        Parameters
        ----------
        block : TextBlockPDF
            Block to classify.

        Returns
        -------
        bool
            Whether the block reads as running text.
        """

        lengths = [len(line.text.strip()) for line in block.lines]
        if not lengths:
            return False
        return median(lengths) > ArticlePDF.MAX_TABLE_CELL_CHARS

    def _get_tables_from_page(self, page_no: int) -> tuple[TablePDF, ...]:
        """
        Internal method to find the tables of a page from their captions.

        A table is found through its caption, the way a figure is. Which side of
        the caption the body is printed on is not fixed — both conventions are in
        use — so it is decided per caption: the body is on the side where the
        nearest block is closer. The body then grows away from the caption, block
        by block, for as long as the blocks follow each other closely and read as
        cells rather than as running text.

        Note that the side is decided on distance alone, deliberately:
        `paragraph_width` is computed with the tables excluded, so consulting it
        here would be circular.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[TablePDF, ...]
            Tables of the page.
        """

        captions = self._table_captions_cache[page_no]
        if not captions:
            return ()

        max_gap = self.MAX_CAPTION_DISTANCE * self.line_pitch
        caption_rects = [caption.rect for caption in captions]

        result: list[TablePDF] = []
        for caption in captions:
            blocks = [
                block
                for block in self._text_cache[page_no]
                if not block.rect.intersects(caption.rect)
                # A table is printed in the column of its caption.
                and block.rect.x1 > caption.rect.x0
                and block.rect.x0 < caption.rect.x1
                # Page furniture is not part of anything: what repeats at the
                # same place on many pages, and whatever sits in the footer.
                and not self._is_running_matter(block.rect)
                and not (
                    not self.footer_rect.is_empty
                    and block.rect.y0 >= self.footer_rect.y0
                )
                and not any(
                    block.rect.intersects(rect)
                    for rect in caption_rects
                    if rect != caption.rect
                )
            ]
            above = sorted(
                (b for b in blocks if b.rect.y1 <= caption.rect.y0),
                key=lambda b: -b.rect.y1,
            )
            below = sorted(
                (b for b in blocks if b.rect.y0 >= caption.rect.y1),
                key=lambda b: b.rect.y0,
            )
            gap_above = (
                caption.rect.y0 - above[0].rect.y1 if above else float("inf")
            )
            gap_below = (
                below[0].rect.y0 - caption.rect.y1 if below else float("inf")
            )
            if min(gap_above, gap_below) > max_gap:
                continue

            caption_above = gap_below <= gap_above
            body = self._walk_table_body(
                below if caption_above else above, caption.rect, caption_above, max_gap
            )
            if not body:
                continue

            table_rect = copy(body[0].rect)
            for block in body[1:]:
                table_rect.include_rect(block.rect)
            # The ruling and the shading of a table are drawn, not written, so
            # they lie outside the blocks the body was walked over.
            for rect in chain(
                self.get_drawing_rects(page_no, copy_rects=False),
                self.get_image_rects(page_no, copy_rects=False),
            ):
                overlap = Rect(rect) & table_rect
                if (
                    not overlap.is_empty
                    and overlap.get_area()
                    > self.MIN_TABLE_RULING_OVERLAP * Rect(rect).get_area()
                ):
                    table_rect.include_rect(rect)
            result.append(
                TablePDF(
                    rect=table_rect, caption=caption, caption_above=caption_above
                )
            )
        return tuple(result)

    @classmethod
    def _walk_table_body(
        cls,
        blocks: list[TextBlockPDF],
        caption_rect: Rect,
        caption_above: bool,
        max_gap: float,
    ) -> list[TextBlockPDF]:
        """
        Internal method to collect the blocks of a table body.

        Parameters
        ----------
        blocks : list[TextBlockPDF]
            Blocks on the table's side of the caption, nearest first.
        caption_rect : Rect
            Rectangle of the caption.
        caption_above : bool
            Whether the caption is above the body.
        max_gap : float
            Largest vertical gap between consecutive blocks of one table.

        Returns
        -------
        list[TextBlockPDF]
            Blocks making up the table body.
        """

        body: list[TextBlockPDF] = []
        edge = caption_rect.y1 if caption_above else caption_rect.y0
        for block in blocks:
            gap = (block.rect.y0 - edge) if caption_above else (edge - block.rect.y1)
            # Blocks side by side in one row do not move the edge apart.
            if gap > max_gap:
                break
            if cls._is_prose(block):
                break
            body.append(block)
            edge = (
                max(edge, block.rect.y1) if caption_above else min(edge, block.rect.y0)
            )
        return body

    @staticmethod
    def _gap_between(rect: Rect, other: Rect) -> float:
        """
        Internal method to measure the blank space between two rectangles.

        Parameters
        ----------
        rect : Rect
            One rectangle. It may be degenerate — a hairline.
        other : Rect
            The other rectangle.

        Returns
        -------
        float
            The larger of the horizontal and vertical gaps between them, 0 if they
            touch or overlap.
        """

        return max(
            rect.x0 - other.x1,
            other.x0 - rect.x1,
            rect.y0 - other.y1,
            other.y0 - rect.y1,
            0.0,
        )

    @staticmethod
    def _span(rect: Rect, other: Rect) -> Rect:
        """
        Internal method to get the rectangle spanning two others.

        Unlike `Rect.include_rect`, it takes a degenerate rectangle — a
        hairline — into account.

        Parameters
        ----------
        rect : Rect
            One rectangle.
        other : Rect
            The other rectangle.

        Returns
        -------
        Rect
            The smallest rectangle holding both.
        """

        return Rect(
            min(rect.x0, other.x0),
            min(rect.y0, other.y0),
            max(rect.x1, other.x1),
            max(rect.y1, other.y1),
        )

    @staticmethod
    def _fraction_inside(rect: Rect, bounds: Rect) -> float:
        """
        Internal method to measure how much of a rectangle lies within bounds.

        Parameters
        ----------
        rect : Rect
            Rectangle to measure. A degenerate one — a hairline — lies either
            wholly within or not at all.
        bounds : Rect
            Bounds to measure against.

        Returns
        -------
        float
            Fraction of the area of `rect` inside `bounds`.
        """

        if rect.is_empty:
            inside = (
                bounds.x0 <= rect.x0
                and rect.x1 <= bounds.x1
                and bounds.y0 <= rect.y0
                and rect.y1 <= bounds.y1
            )
            return 1.0 if inside else 0.0
        if not rect.intersects(bounds):
            return 0.0
        return Rect(rect).intersect(bounds).get_area() / rect.get_area()

    def _is_visible_drawing(self, drawing: DrawingObjectPDF) -> bool:
        """
        Internal method to tell whether a drawing shows on the page.

        A drawing painted white or fully transparent is a background, a mask
        or a spacer — often laid under a caption or the whole of a figure —
        and would join whatever lies near it into one figure.

        Parameters
        ----------
        drawing : DrawingObjectPDF
            Drawing to check.

        Returns
        -------
        bool
            Whether the drawing leaves a mark on the page.
        """

        def is_white(color: Sequence[float] | None) -> bool:
            return color is None or all(value >= self.WHITE_LEVEL for value in color)

        stroked = (
            drawing.type in ("s", "fs")
            and drawing.stroke_opacity > 0
            and not is_white(drawing.color)
        )
        filled = (
            drawing.type in ("f", "fs")
            and drawing.fill_opacity > 0
            and not is_white(drawing.fill)
        )
        return stroked or filled

    def _get_figure_graphics(
        self,
        page_no: int,
        caption_rects: Sequence[Rect],
        paragraph_rects: Sequence[Rect],
    ) -> list[Rect]:
        """
        Internal method to get the drawings and images a figure can be made of.

        Left out are what only looks like part of a figure: the page furniture
        in the header and footer bands, a table's ruling, a rule of the page
        (a hairline most of the page wide), decoration bleeding off the page
        edge, a frame drawn around text or a border beside a caption, anything
        invisible (see `_is_visible_drawing`) and a background lying mostly
        under a caption.
        An image whose blank margin runs under its caption is cut short at the
        caption instead.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).
        caption_rects : Sequence[Rect]
            Figure and table captions of the page.
        paragraph_rects : Sequence[Rect]
            Paragraphs of the page.

        Returns
        -------
        list[Rect]
            Rectangles of the drawings and images, as copies.
        """

        page_rect = cast(Rect, self._page(page_no).rect)
        page_area = page_rect.get_area()
        table_rects = self.get_table_rects(page_no, copy_rects=False)
        # Each candidate goes with the width of the stroke it is painted with:
        # a line of a plot can be drawn thicker than its path is wide.
        candidates = [
            (copy(drawing.rect), drawing.width if drawing.type in ("s", "fs") else 0.0)
            for drawing in self._drawings_cache[page_no]
            if self._is_visible_drawing(drawing)
        ]
        candidates += [(rect, 0.0) for rect in self.get_image_rects(page_no)]

        result: list[Rect] = []
        for rect, stroke_width in candidates:
            if rect.get_area() > page_area * self.MAX_IMAGE_AREA:
                continue
            if (
                rect.x0 < page_rect.x0 - self.MARGIN
                or rect.y0 < page_rect.y0 - self.MARGIN
                or rect.x1 > page_rect.x1 + self.MARGIN
                or rect.y1 > page_rect.y1 + self.MARGIN
            ):
                continue
            if (
                max(min(rect.width, rect.height), stroke_width) < self.MAX_HAIRLINE_WIDTH
                and max(rect.width, rect.height)
                > page_rect.width * self.MIN_PAGE_RULE_LENGTH
            ):
                continue
            if not self.header_rect.is_empty and rect.y1 <= self.header_rect.y1 + self.MARGIN:
                continue
            if not self.footer_rect.is_empty and rect.y0 >= self.footer_rect.y0 - self.MARGIN:
                continue
            if any(
                self._fraction_inside(rect, table) > self.MIN_TABLE_RULING_OVERLAP
                for table in table_rects
            ):
                continue
            if any(rect.contains(text) for text in chain(caption_rects, paragraph_rects)):
                continue
            if min(rect.width, rect.height) < self.MAX_HAIRLINE_WIDTH and any(
                self._is_beside(rect, caption)
                and self._gap_between(rect, caption) <= self.FIGURE_GRAPHICS_GAP
                for caption in caption_rects
            ):
                continue
            if any(
                self._fraction_inside(rect, caption) > 0.5 for caption in caption_rects
            ):
                continue
            for caption in caption_rects:
                if not rect.intersects(caption):
                    continue
                if rect.y0 + rect.y1 < caption.y0 + caption.y1:
                    rect.y1 = min(rect.y1, caption.y0)
                else:
                    rect.y0 = max(rect.y0, caption.y1)
            result.append(rect)
        return result

    def _column_gutter_middle(self, page_no: int) -> float | None:
        """
        Internal method to locate the middle of the column gutter of a page.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        float | None
            Middle of the gap between the right edge of the left column and
            the left edge of the right column, or None unless the document is
            set in two columns and the page carries paragraphs in both.
        """

        if self.columns_number != 2:
            return None
        page_width = cast(Rect, self._page(page_no).rect).width
        paragraph_rects = self.get_paragraph_rects(page_no, copy_rects=False)
        left_columns_x1 = [rect.x1 for rect in paragraph_rects if rect.x1 < page_width * 0.6]
        right_columns_x0 = [rect.x0 for rect in paragraph_rects if rect.x0 > page_width * 0.4]
        if not left_columns_x1 or not right_columns_x0:
            return None
        return (max(left_columns_x1) + min(right_columns_x0)) / 2

    def _is_figure_label(
        self,
        block: TextBlockPDF,
        caption_rects: Sequence[Rect],
        paragraph_rects: Sequence[Rect],
    ) -> bool:
        """
        Internal method to tell whether a text block could be set in a figure.

        Axis labels, legends and panel letters are short: running text is not,
        and neither is text belonging to a paragraph or a caption, nor the page
        furniture — a running head can be merged with what follows it into one
        block reaching below the header band. An axis title can be as long as a
        line of running text, but it is a single line set in a typeface other
        than the body's.

        Parameters
        ----------
        block : TextBlockPDF
            Block to check.
        caption_rects : Sequence[Rect]
            Figure and table captions of the page.
        paragraph_rects : Sequence[Rect]
            Paragraphs of the page.

        Returns
        -------
        bool
            Whether the block can be a label of a figure.
        """

        rect = block.rect
        if self._is_prose(block) and (
            len(block.lines) > 1 or self._is_body_text(block)
        ):
            return False
        if rect.intersects(self.header_rect) or rect.intersects(self.footer_rect):
            return False
        if self._is_running_matter(rect):
            return False
        if any(rect.intersects(caption) for caption in caption_rects):
            return False
        return not any(
            self._fraction_inside(rect, paragraph) > 0.5 for paragraph in paragraph_rects
        )

    def _merge_clusters(
        self, clusters: Iterable[FigureClusterPDF]
    ) -> list[FigureClusterPDF]:
        """
        Internal method to merge clusters lying within `FIGURE_GRAPHICS_GAP` of each other.

        Parameters
        ----------
        clusters : Iterable[FigureClusterPDF]
            Clusters to merge. They are merged in place.

        Returns
        -------
        list[FigureClusterPDF]
            The clusters left, no two of them within reach of each other.
        """

        result = list(clusters)
        merged = True
        while merged:
            merged = False
            kept: list[FigureClusterPDF] = []
            for cluster in sorted(result, key=lambda c: (c.reach.y0, c.reach.x0)):
                for other in kept:
                    if (
                        self._gap_between(other.reach, cluster.reach)
                        <= self.FIGURE_GRAPHICS_GAP
                    ):
                        other.reach = self._span(other.reach, cluster.reach)
                        other.rect.include_rect(cluster.rect)
                        merged = True
                        break
                else:
                    kept.append(cluster)
            result = kept
        return result

    def _get_figure_clusters_from_page(
        self, page_no: int
    ) -> tuple[FigureClusterPDF, ...]:
        """
        Internal method to group the graphics of a page into clusters.

        Graphics (see `_get_figure_graphics`) lying within `FIGURE_GRAPHICS_GAP`
        of each other are one cluster. A cluster then takes in the text set
        within `FIGURE_LABEL_GAP` of it that can be a label (see
        `_is_figure_label`), except for text in the body font above it — the
        heading of the section the figure opens — and, on a two-column page,
        text across the column gutter from a cluster confined to one column: a
        panel letter of the figure in the next column can lie nearer to this
        one than to its own. Clusters too small to be a figure are dropped.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[FigureClusterPDF, ...]
            Clusters of the page.
        """

        caption_rects = [
            *self.get_figure_caption_rects(page_no, copy_rects=False),
            *self.get_table_caption_rects(page_no, copy_rects=False),
        ]
        paragraph_rects = self.get_paragraph_rects(page_no, copy_rects=False)
        clusters = self._merge_clusters(
            FigureClusterPDF(reach=copy(rect), rect=copy(rect))
            for rect in self._get_figure_graphics(page_no, caption_rects, paragraph_rects)
        )

        gutter = self._column_gutter_middle(page_no)
        labels = [
            block
            for block in self._text_cache[page_no]
            if self._is_figure_label(block, caption_rects, paragraph_rects)
        ]
        # A label can bring a cluster within reach of the next label, or of
        # another cluster: a few rounds settle it.
        for _ in range(3):
            added = False
            for block in labels:
                rect = block.rect
                for cluster in clusters:
                    if cluster.reach.contains(rect):
                        cluster.rect.include_rect(rect)
                        break
                    if self._gap_between(cluster.reach, rect) > self.FIGURE_LABEL_GAP:
                        continue
                    if gutter is not None and not (
                        cluster.reach.x0 < gutter < cluster.reach.x1
                    ):
                        in_left_column = cluster.reach.x1 <= gutter
                        if in_left_column != (rect.x0 + rect.x1 <= 2 * gutter):
                            continue
                    if self._is_body_text(block) and rect.y1 <= cluster.reach.y0:
                        continue
                    cluster.reach = self._span(cluster.reach, rect)
                    cluster.rect.include_rect(rect)
                    added = True
                    break
            clusters = self._merge_clusters(clusters)
            if not added:
                break

        result: list[FigureClusterPDF] = []
        for cluster in clusters:
            reach = cluster.reach
            if max(reach.width, reach.height) < self.MIN_FIGURE_SIDE:
                continue
            if min(reach.width, reach.height) < self.MAX_HAIRLINE_WIDTH:
                continue
            if cluster.rect.is_empty:
                cluster.rect = copy(reach)
            result.append(cluster)
        return tuple(result)

    def _caption_distance(self, figure_rect: Rect, caption_rect: Rect) -> float | None:
        """
        Internal method to measure how far a caption is from a figure it could belong to.

        Parameters
        ----------
        figure_rect : Rect
            Rectangle of the figure.
        caption_rect : Rect
            Rectangle of the caption.

        Returns
        -------
        float | None
            The blank space between them, weighted by how often a caption is set
            that way round (see `CAPTION_ABOVE_FIGURE_WEIGHT`,
            `SIDE_CAPTION_WEIGHT`), or None if the caption is neither above,
            below nor beside the figure.
        """

        overlaps_horizontally = min(figure_rect.x1, caption_rect.x1) > max(
            figure_rect.x0, caption_rect.x0
        )
        if overlaps_horizontally:
            if figure_rect.y1 <= caption_rect.y0 + self.MAX_CAPTION_FIGURE_OVERLAP:
                return max(caption_rect.y0 - figure_rect.y1, 0.0)
            if figure_rect.y0 >= caption_rect.y1 - self.MAX_CAPTION_FIGURE_OVERLAP:
                return (
                    max(figure_rect.y0 - caption_rect.y1, 0.0)
                    * self.CAPTION_ABOVE_FIGURE_WEIGHT
                )
        if self._is_beside(figure_rect, caption_rect):
            return (
                max(caption_rect.x0 - figure_rect.x1, figure_rect.x0 - caption_rect.x1, 0.0)
                * self.SIDE_CAPTION_WEIGHT
            )
        return None

    @staticmethod
    def _is_beside(figure_rect: Rect, caption_rect: Rect) -> bool:
        """
        Internal method to tell whether a caption is set beside a figure.

        Parameters
        ----------
        figure_rect : Rect
            Rectangle of the figure.
        caption_rect : Rect
            Rectangle of the caption.

        Returns
        -------
        bool
            Whether the two share most of the height of the shorter one.
        """

        shared_height = min(figure_rect.y1, caption_rect.y1) - max(
            figure_rect.y0, caption_rect.y0
        )
        return shared_height > 0.5 * min(figure_rect.height, caption_rect.height)

    @staticmethod
    def _is_separated(rect: Rect, other: Rect, obstacles: Iterable[Rect]) -> bool:
        """
        Internal method to tell whether text stands between two rectangles.

        Parameters
        ----------
        rect : Rect
            One rectangle.
        other : Rect
            The other rectangle.
        obstacles : Iterable[Rect]
            Paragraphs and captions that can stand between them.

        Returns
        -------
        bool
            Whether most of an obstacle touching neither lies in the rectangle
            spanning both.
        """

        span = ArticlePDF._span(rect, other)
        for obstacle in obstacles:
            if obstacle.intersects(rect) or obstacle.intersects(other):
                continue
            if ArticlePDF._fraction_inside(obstacle, span) > 0.5:
                return True
        return False

    def _match_figures_on_page(self, page_no: int) -> dict[int, Rect]:
        """
        Internal method to pair the figure captions of a page with its graphics.

        Every caption takes a cluster of graphics above, below or beside it
        (see `_caption_distance`) that no paragraph or caption separates from
        it. The pairs are chosen together rather than nearest first: as many
        captions as possible get a figure, and among those pairings the one
        with the least distance in total wins. A caption set between two
        figures lies nearer to the figure it does not belong to often enough —
        the next figure starts right under it — and taking the nearest would
        leave the other caption with none. Each figure then takes in the clusters no
        caption claimed that lie within `MAX_FIGURE_PANEL_GAP` of it, provided it
        covers no text, other figure or page furniture in doing so — the panels
        of a figure set further apart than `FIGURE_GRAPHICS_GAP`. A caption
        whose figure sits at the foot of the previous page takes part in
        neither.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        dict[int, Rect]
            Figure rectangles keyed by the index of their caption among the
            figure captions of the page. A caption no cluster answers to is left
            out.
        """

        captions = self._figure_captions_cache[page_no]
        clusters = self._figure_clusters_cache[page_no]
        obstacles = [
            *self.get_paragraph_rects(page_no, copy_rects=False),
            *self.get_figure_caption_rects(page_no, copy_rects=False),
            *self.get_table_caption_rects(page_no, copy_rects=False),
        ]

        # A pair that cannot be made costs more than every pair that can put
        # together, so that as many captions as possible find a figure before
        # the distances are weighed at all.
        unmatched = (len(captions) + 1) * self.MAX_FIGURE_CAPTION_GAP * max(
            1.0, self.CAPTION_ABOVE_FIGURE_WEIGHT, self.SIDE_CAPTION_WEIGHT
        )
        costs = np.full((len(captions), len(clusters)), unmatched)
        for caption_index, caption in enumerate(captions):
            if self._figure_rect_on_previous_page(page_no, caption) is not None:
                continue
            for cluster_index, cluster in enumerate(clusters):
                distance = self._caption_distance(cluster.reach, caption.rect)
                if distance is None or distance > self.MAX_FIGURE_CAPTION_GAP:
                    continue
                if self._is_separated(cluster.reach, caption.rect, obstacles):
                    continue
                costs[caption_index, cluster_index] = distance

        figures: dict[int, FigureClusterPDF] = {}
        claimed: set[int] = set()
        for caption_index, cluster_index in zip(*linear_sum_assignment(costs)):
            if costs[caption_index, cluster_index] >= unmatched:
                continue
            cluster = clusters[cluster_index]
            figures[int(caption_index)] = FigureClusterPDF(
                reach=copy(cluster.reach), rect=copy(cluster.rect)
            )
            claimed.add(int(cluster_index))

        grown = True
        while grown:
            grown = False
            for cluster_index, cluster in enumerate(clusters):
                if cluster_index in claimed:
                    continue
                if cluster.reach.intersects(self.header_rect) or cluster.reach.intersects(
                    self.footer_rect
                ):
                    continue
                nearest: tuple[float, int] | None = None
                for caption_index, figure in figures.items():
                    gap = self._gap_between(figure.reach, cluster.reach)
                    if gap > self.MAX_FIGURE_PANEL_GAP:
                        continue
                    span = self._span(figure.reach, cluster.reach)
                    if any(
                        self._fraction_inside(obstacle, span) > self.MAX_FIGURE_TEXT_OVERLAP
                        for obstacle in obstacles
                    ):
                        continue
                    if any(
                        other is not figure and span.intersects(other.reach)
                        for other in figures.values()
                    ):
                        continue
                    if nearest is None or gap < nearest[0]:
                        nearest = (gap, caption_index)
                if nearest is not None:
                    figure = figures[nearest[1]]
                    figure.reach = self._span(figure.reach, cluster.reach)
                    figure.rect.include_rect(cluster.rect)
                    claimed.add(cluster_index)
                    grown = True

        return {index: figure.rect for index, figure in figures.items()}

    def _get_figures_from_page(
        self,
        page_no: int,
    ) -> tuple[FigurePDF, ...]:
        """
        Internal method to get the figures of a single page.

        A figure is found from what it is made of: the clusters of graphics on
        the page are paired with the captions next to them (see
        `_match_figures_on_page`), whichever side of the figure the caption is
        set on. A caption no cluster answers to — on a scanned page, or under a
        figure drawn in text — is bounded from the caption instead
        (`_bound_figure_rect`): its lower bound is the caption, its upper bound
        the nearest paragraph, caption, figure or header above it, and its
        sides are narrowed to the caption's column in a two-column document.
        A figure whose caption was carried over to the top of the next page is
        stored on this page, where it is drawn.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).

        Returns
        -------
        tuple[FigurePDF, ...]
            Figures drawn on the page.
        """

        page = self._page(page_no)
        figure_captions = self._figure_captions_cache[page_no]
        matches = self._figure_matches_cache[page_no]
        matched = {
            index: FigurePDF(copy(rect), figure_captions[index])
            for index, rect in matches.items()
        }

        result: list[FigurePDF] = []
        for index, caption in enumerate(figure_captions):
            if index in matched:
                result.append(matched[index])
                continue
            # The figure was printed at the foot of the previous page, and is
            # bounded there.
            if self._figure_rect_on_previous_page(page_no, caption) is not None:
                continue
            if self.columns_number > 2:
                warn(
                    f"Document {page.parent} page {page.number} has more than 2 columns. "
                    "Only 2 columns are supported for figures bounded from their caption."
                )
                continue

            figure_rect, gutter = self._bound_figure_rect(
                page_no, caption, list(matched.values()), full_width=False
            )
            # A figure narrowed to one column that nevertheless holds an element
            # running across the gutter spans both columns after all — its caption
            # was merely too short to say so. Bound it again as a full-width
            # figure, so that its upper bound clears the paragraphs of both
            # columns rather than of one.
            if gutter is not None and self._crosses_column_gutter(
                page_no, figure_rect, gutter
            ):
                figure_rect, _ = self._bound_figure_rect(
                    page_no, caption, list(matched.values()), full_width=True
                )
            result.append(
                FigurePDF(self._refine_figure_rect(page_no, figure_rect), caption)
            )

        if page_no + 1 < self.file.page_count:
            for caption in self._figure_captions_cache[page_no + 1]:
                prev_page_rect = self._figure_rect_on_previous_page(
                    page_no + 1, caption
                )
                if prev_page_rect is not None:
                    result.append(
                        FigurePDF(
                            self._refine_figure_rect(page_no, prev_page_rect), caption
                        )
                    )
        return tuple(result)

    def _figure_rect_on_previous_page(
        self,
        page_no: int,
        caption: FigureCaptionPDF,
    ) -> Rect | None:
        """
        Bound the figure of a caption that opens its page, if it sits on the page before.

        A word processor that cannot fit a figure together with its caption
        leaves the figure at the foot of one page and carries the caption over
        to the top of the next. Such a caption has nothing above it but the
        running head, and the previous page ends in graphics below its last
        paragraph or caption. A caption with graphics beside it is a side
        caption at the top of its page instead, and belongs to them.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed) the caption sits on.
        caption : FigureCaptionPDF
            Caption of the figure.

        Returns
        -------
        Rect | None
            Maximum bounds of the figure on page `page_no - 1`, or None if the
            figure is on the caption's own page.
        """

        if page_no == 0:
            return None
        if any(
            self._is_beside(cluster.reach, caption.rect)
            for cluster in self._figure_clusters_cache[page_no]
        ):
            return None

        page_rect = cast(Rect, self._page(page_no).rect)
        above_caption = Rect(
            page_rect.x0, self.header_rect.y1, page_rect.x1, caption.rect.y0
        )
        for rect in chain(
            self.get_text_rects(page_no, copy_rects=False),
            self.get_drawing_rects(page_no, copy_rects=False),
            self.get_image_rects(page_no, copy_rects=False),
        ):
            if above_caption.contains(rect) and not self._is_running_matter(rect):
                return None

        prev_page_no = page_no - 1
        figure_rect = copy(
            cast(Rect, self._page(prev_page_no).rect)
        )
        figure_rect.x0, figure_rect.x1 = self._line_number_gutter(prev_page_no)
        if not self.footer_rect.is_empty:
            figure_rect.y1 = self.footer_rect.y0
        figure_rect.y0 = max(
            (
                rect.y1
                for rect in chain(
                    [self.header_rect],
                    self.get_paragraph_rects(prev_page_no, copy_rects=False),
                    self.get_figure_caption_rects(prev_page_no, copy_rects=False),
                    self.get_table_caption_rects(prev_page_no, copy_rects=False),
                    self.get_table_rects(prev_page_no, copy_rects=False),
                )
            ),
        )
        if figure_rect.is_empty:
            return None
        if not self.get_drawing_rects(
            prev_page_no, clip=figure_rect, copy_rects=False
        ) and not self.get_image_rects(prev_page_no, clip=figure_rect, copy_rects=False):
            return None
        return figure_rect

    def _bound_figure_rect(
        self,
        page_no: int,
        caption: FigureCaptionPDF,
        other_figures: Sequence[FigurePDF],
        full_width: bool,
    ) -> tuple[Rect, tuple[float, float] | None]:
        """
        Bound the figure belonging to one caption.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed) the caption sits on.
        caption : FigureCaptionPDF
            Caption of the figure to bound.
        other_figures : Sequence[FigurePDF]
            Figures of the page already bounded from their graphics. Those above
            the caption and over part of its width take part as upper-bound
            candidates.
        full_width : bool
            Bound the figure as spanning every column: neither the upper bound
            nor the sides are restricted to the caption's own column.

        Returns
        -------
        tuple[Rect, tuple[float, float] | None]
            The figure rectangle, and the gutter the sides were narrowed to as
            `(right edge of the left column, left edge of the right column)`.
            The gutter is None whenever the sides were left alone — a
            single-column document, `full_width`, a caption spanning both
            columns, or a page carrying only one column of paragraphs.
        """

        page = self._page(page_no)
        page_width = page.rect.width
        figure_rect = cast(Rect, copy(page.rect))
        figure_rect.x0, figure_rect.x1 = self._line_number_gutter(page_no)

        # Lower boundary
        figure_rect.y1 = caption.rect.y0

        # Upper boundary
        upper_bound_candidates: list[Rect] = []
        paragraphs_above = self.get_paragraph_rects(page_no=page_no, clip=figure_rect)

        # Columns could be shifted horizontaly so that the right boundary of
        # the left column can be located more to the right than half of page width
        # and vice versa
        min_x0_right_col = page_width * 0.4
        max_x1_left_col = page_width * 0.6
        in_left_column = caption.rect.x1 < max_x1_left_col
        in_right_column = caption.rect.x0 > min_x0_right_col
        by_column = self.columns_number > 1 and not full_width

        # if we have two column layout we need to check if figure is
        # located in particular column and if so, check only paragraphs
        # in that column
        if by_column:
            # figure is in right column
            if in_right_column:
                paragraphs_above = [
                    rect for rect in paragraphs_above if rect.x0 > min_x0_right_col
                ]
            # Figure is in left column
            if in_left_column:
                paragraphs_above = [
                    rect for rect in paragraphs_above if rect.x1 < max_x1_left_col
                ]

        if paragraphs_above:
            lowest_paragraph = max(paragraphs_above, key=lambda x: x.y1)
            upper_bound_candidates.append(lowest_paragraph)

        # Search for figure caption above
        other_caption_rects = self.get_figure_caption_rects(page_no, clip=figure_rect)
        if by_column:
            # If figure is in right column, then we don't care
            # about other figures in left column
            if in_right_column:
                other_caption_rects = [
                    rect for rect in other_caption_rects if rect.x0 > min_x0_right_col
                ]
            # Figure is in left column, then we don't care
            # about other figures
            elif in_left_column:
                other_caption_rects = [
                    rect for rect in other_caption_rects if rect.x1 < max_x1_left_col
                ]

        if other_caption_rects:
            lowest_caption = max(other_caption_rects, key=lambda x: x.y1)
            upper_bound_candidates.append(lowest_caption)

        upper_bound_candidates.append(self.header_rect)
        upper_bound_candidates.extend(
            [
                figure.rect
                for figure in other_figures
                if figure.rect.y1 < caption.rect.y0
                and figure.rect.x0 < caption.rect.x1
                and caption.rect.x0 < figure.rect.x1
            ]
        )

        if upper_bound_candidates:
            lowest_bounding_rect = max(upper_bound_candidates, key=lambda x: x.y1)
            figure_rect.y0 = lowest_bounding_rect.y1

        # Side boundaries
        gutter: tuple[float, float] | None = None
        if by_column:
            paragraph_rects = self.get_paragraph_rects(page_no, copy_rects=False)
            left_columns_x1 = [
                rect.x1 for rect in paragraph_rects if rect.x1 < max_x1_left_col
            ]
            right_columns_x0 = [
                rect.x0 for rect in paragraph_rects if rect.x0 > min_x0_right_col
            ]
            narrowed = False

            # Left boundary for figure in right column
            if in_right_column:
                # First try to set it slightly to the right from the most right
                # position of the left column. But left column could be missing
                if left_columns_x1:
                    figure_rect.x0 = max(left_columns_x1) + self.MARGIN
                    narrowed = True
                elif right_columns_x0:
                    figure_rect.x0 = max(right_columns_x0) - self.MARGIN
                    narrowed = True

            # Right boundary for figure in left column
            if in_left_column:
                if right_columns_x0:
                    figure_rect.x1 = min(right_columns_x0) - self.MARGIN
                    narrowed = True
                elif left_columns_x1:
                    figure_rect.x1 = min(left_columns_x1) + self.MARGIN
                    narrowed = True

            if narrowed and left_columns_x1 and right_columns_x0:
                gutter = (max(left_columns_x1), min(right_columns_x0))

        return figure_rect, gutter

    def _crosses_column_gutter(
        self,
        page_no: int,
        figure_rect: Rect,
        gutter: tuple[float, float],
    ) -> bool:
        """
        Check whether anything beside the figure runs across the column gutter.

        The gutter is measured from the paragraphs themselves rather than taken
        as a fraction of the page: the thresholds separating the columns leave a
        fifth of the page between them, and a narrow element parked in that band
        — a panel letter, the rule under it — sits in one column while reading as
        if it spanned both.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).
        figure_rect : Rect
            Figure rectangle, narrowed to its column. Only its vertical span is
            used: the element that gives the figure away lies outside its sides,
            which is why it was not caught when the sides were set.
        gutter : tuple[float, float]
            Right edge of the left column and left edge of the right column.

        Returns
        -------
        bool
            True if an element of the page crosses the gutter within the
            figure's vertical span.
        """

        gutter_left, gutter_right = gutter
        for rect in chain(
            self.get_text_rects(page_no, copy_rects=False),
            self.get_drawing_rects(page_no, copy_rects=False),
            self.get_image_rects(page_no, copy_rects=False),
        ):
            if rect.y1 <= figure_rect.y0 or rect.y0 >= figure_rect.y1:
                continue
            # Page furniture crosses the gutter by construction: a running head
            # is one block the width of the text area, and it reaches below
            # `header_rect`, which is a band of the page rather than the block.
            if rect.intersects(self.header_rect) or rect.intersects(self.footer_rect):
                continue
            if self._is_running_matter(rect):
                continue
            if rect.x0 < gutter_left and rect.x1 > gutter_right:
                return True
        return False

    def _refine_figure_rect(
        self,
        page_no: int,
        figure_rect: Rect,
    ) -> Rect:

        """
        Shrink a figure rectangle to the elements it holds.

        Text set in the body font above every image and drawing of the figure is
        left out: it is document text — a section heading or a short list with
        no paragraph between it and the figure — that the upper bound, drawn at
        the nearest paragraph, failed to exclude. The labels of a figure are
        either set in a typeface of their own or sit among its graphics.

        An image belongs to the figure when most of it lies within `figure_rect`
        and only its top sticks out, not only when all of it does:
        an image placed by a word processor can reach a few points up past the
        heading above it, the bound the rectangle stops at, and would otherwise
        be lost to the figure.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed) the figure sits on.
        figure_rect : Rect
            Maximum bounds of the figure.

        Returns
        -------
        Rect
            The rectangle enclosing the elements of the figure, or `figure_rect`
            itself if it holds none.
        """

        # An image may rise above the top of the figure, never cross its sides
        # or its caption beneath. Drawings must lie wholly within: the rules of
        # a frame or table around figure and caption alike are no part of it.
        top_free_rect = Rect(figure_rect.x0, -inf, figure_rect.x1, figure_rect.y1)
        graphics = self.get_drawing_rects(page_no, clip=figure_rect, copy_rects=False)
        graphics += [
            rect
            for rect in self.get_image_rects(page_no, copy_rects=False)
            if figure_rect.contains(rect)
            or (
                top_free_rect.contains(rect)
                and not rect.is_empty
                and Rect(rect).intersect(figure_rect).get_area() / rect.get_area()
                >= self.MIN_FIGURE_IMAGE_OVERLAP
            )
        ]
        graphics_top = min((rect.y0 for rect in graphics), default=figure_rect.y1)
        graphics_left = min((rect.x0 for rect in graphics), default=figure_rect.x0)
        graphics_right = max((rect.x1 for rect in graphics), default=figure_rect.x1)

        rects: list[Rect] = list(graphics)
        for block in self._text_cache[page_no]:
            if not figure_rect.contains(block.rect):
                continue
            if graphics and self._is_body_text(block):
                if block.rect.y1 <= graphics_top:
                    continue
                # Running text beside the graphics is the column wrapped around
                # a figure wider than one column but narrower than the page.
                beside = (
                    block.rect.x1 <= graphics_left or block.rect.x0 >= graphics_right
                )
                if beside and self._is_prose(block):
                    continue
            rects.append(block.rect)
        if not rects:
            return figure_rect
        refined_rect = copy(rects[0])
        for rect in rects:
            refined_rect.include_rect(rect)
        return refined_rect
