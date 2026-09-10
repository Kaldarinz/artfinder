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
from collections.abc import Iterable
from copy import copy, deepcopy
from functools import cached_property
from itertools import chain
from math import floor
from statistics import median
from os import PathLike
from pathlib import Path
from typing import Literal, cast
from warnings import warn

import pandas as pd
import pymupdf
from pymupdf import Page, Pixmap, Rect
from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]

from artfinder.dataclasses import (
    DocumentElementsPDF,
    DrawingObjectPDF,
    FigureCaptionPDF,
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
    """How far the distance between two lines may differ from `line_pitch`, as a
    fraction of it, for them to still be consecutive lines of one caption."""
    DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
    "Pattern of a DOI in text. The prefix dot is literal: `1000/x` is not a DOI."
    CAPTION_PATTERN = re.compile(
        r"^\s*(?:(?P<supp>supplementary|supplemental|supporting)\s+)?"
        r"Fig(?:\.|ure\.?|)\s+(?:(?P<prefix>S)\s?)?(?P<number>\d+)\W*",
        re.IGNORECASE,
    )
    """Pattern of a figure caption opening, e.g. `Fig. 1`, `Figure S2.` or
    `Supplementary Figure 3`. The `supp` and `prefix` groups mark a
    supplementary figure, the `number` group holds its digits."""
    TABLE_CAPTION_PATTERN = re.compile(
        r"^\s*(?:(?P<supp>supplementary|supplemental|supporting)\s+)?"
        r"Table\s+(?:(?P<prefix>S)\s?)?(?P<number>\d+)\W*",
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
                self.identifier = self.make_valid_filename(self.file[0].get_text()[:50])  # type: ignore

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

        Measured as the median distance inside multi-line text blocks, so it
        reflects the leading the document is actually set with. This is not the
        same as the height of a line: a manuscript set with extra leading spaces
        its lines further apart than their glyphs are tall, and a caption
        continued on the next line is then not one line height below its
        predecessor but one pitch.

        Returns
        -------
        float
            Median distance between consecutive lines, in points. Falls back to
            the median line height if the document has no multi-line block.
        """

        gaps: list[float] = []
        heights: list[float] = []
        for page_no in range(self.file.page_count):
            for block in self._text_cache[page_no]:
                heights.extend(line.rect.height for line in block.lines)
                for line, next_line in zip(block.lines, block.lines[1:]):
                    gap = next_line.rect.y1 - line.rect.y1
                    if 0 < gap < self.MAX_LINE_PITCH:
                        gaps.append(gap)
        if gaps:
            return median(gaps)
        return median(heights) if heights else 0.0

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
        all_widths = [self.file[i].rect.width for i in range(self.file.page_count)]
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
        search_rect = cast(Rect, self.file[0].rect)
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
        caption, a figure with a side caption — something to stop against.

        Returns
        -------
        Rect
            The footer rectangle if found, otherwise an empty Rect.
        """

        page_rect = cast(Rect, self.file[0].rect)  # type: ignore[attr-defined]
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

        if isinstance(page_no, Page):  # type: ignore[arg-type]
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

        A line joins the paragraph above when it follows one `line_pitch` below
        it — the rule the captions are stitched by — and lies within its
        horizontal span.

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
        paragraphs = [
            block
            for block in blocks
            if self.paragraph_width.min <= block.rect.width <= self.paragraph_width.max
        ]
        rects = [copy(block.rect) for block in paragraphs]
        taken = {id(block) for block in paragraphs}
        tolerance = max(1.0, self.LINE_PITCH_TOLERANCE * self.line_pitch)

        extended = True
        while extended:
            extended = False
            for rect in rects:
                for block in blocks:
                    if id(block) in taken or not block.lines:
                        continue
                    first_line = block.lines[0].rect
                    if abs(first_line.y1 - rect.y1 - self.line_pitch) > tolerance:
                        continue
                    overlap = min(rect.x1, block.rect.x1) - max(rect.x0, block.rect.x0)
                    if overlap < self.MIN_PARAGRAPH_OVERLAP * block.rect.width:
                        continue
                    rect.include_rect(block.rect)
                    taken.add(id(block))
                    extended = True
        return tuple(rects)

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
        page_no: Page | int | None = None,  # type: ignore[valid-type]
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
        page_no: Page | int | None = None,  # type: ignore[valid-type]
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
        page_no: Page | int | None = None,  # type: ignore[valid-type]
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

        if isinstance(page_no, Page):  # type: ignore[arg-type]
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
            page_text = self.file[page_no].get_text()
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

        An image reported by `get_image_info` is measured over its whole
        placement box, so a cropped one reads lower than its true resolution —
        the estimate errs low.

        Parameters
        ----------
        figure_label : str
            Label of the figure.

        Returns
        -------
        int
            DPI to rasterize the figure at.
        """

        dpis = [
            max(
                image.width * self.POINTS_PER_INCH / image.rect.width,
                image.height * self.POINTS_PER_INCH / image.rect.height,
            )
            for image in self.get_figure_images(figure_label)
            if image.width
            and image.height
            and image.rect.width > 0
            and image.rect.height > 0
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
            pixmap = self.file[page_ind].get_pixmap(
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
            pages = cast(list[Page], list(self.file))  # type: ignore
        else:
            pages = [self.file[page_number]]

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
                        self.get_table_rects(page.number, copy_rects=False)  # type: ignore[attr-defined]
                    )
                if element in [
                    DocumentElementsPDF.ALL,
                    DocumentElementsPDF.TABLE_CAPTION,
                ]:
                    rects.extend(
                        self.get_table_caption_rects(page.number, copy_rects=False)  # type: ignore[attr-defined]
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
            TextBlockPDF.from_dict(block)  # type: ignore
            for block in self.file[page_no].get_text(option="dict")["blocks"]  # type: ignore
            if block["type"] == 0  # type: ignore
        ]
        for block in text_blocks:
            # Skip blank blocks
            if len(block.text) == 0 or block.text.isspace():
                continue
            result.append(block)
        return tuple(result)

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
            page_width = cast(Rect, self.file[page_no].rect).width  # type: ignore
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

        page_rect = cast(Rect, self.file[page_no].rect)  # type: ignore
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

        page = self.file[page_no]
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

        page = self.file[page_no]
        page_area = abs(page.rect)
        all_images = tuple(
            [
                ImageInfoPDF.from_dict(info)
                for info in page.get_image_info(hashes=True, xrefs=True)
            ]
        )
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
                width=self.file[0].rect.width, height=self.file[0].rect.height
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
                width=self.file[0].rect.width, height=self.file[0].rect.height
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
                width=self.file[0].rect.width, height=self.file[0].rect.height
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

        Parameters
        ----------
        eps : float
            The maximum distance between two samples for one to be considered as
            in the neighborhood of the other.

        Returns
        -------
        Size
            Mean, min, max width of the paragraph-width cluster.
        """

        rects: list[Rect] = [
            block.rect
            for page_no in range(self.file.page_count)
            for block in self._text_cache[page_no]
            if self._is_prose(block)
        ]
        if not rects:
            warn(f"No running text found in {self}")
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
            warn(f"No clusters found (all points are noise) in {self}")
            return Size(mean=0.0, min=0.0, max=0.0)

        clusters = valid_blocks.pivot_table(
            index="groups",
            aggfunc={"width": ["mean", "min", "max", "count"], "height": "sum"},  # type: ignore
        ).sort_values(by=("height", "sum"), ascending=False)

        largest_cluster = clusters.iloc[0]
        width_mean = largest_cluster[("width", "mean")]
        width_min = largest_cluster[("width", "min")]
        width_max = largest_cluster[("width", "max")]
        return Size(mean=width_mean, min=width_min, max=width_max)  # type: ignore

    @staticmethod
    def _caption_label(match: re.Match[str]) -> str:
        """
        Build a figure label from a `CAPTION_PATTERN` match.

        The label is the figure number with leading zeros dropped, prefixed with
        `S` for a supplementary figure. Both spellings of "supplementary" mark
        the same thing, so `Supplementary Fig. S1` is labelled `S1`, not `SS1`.

        Parameters
        ----------
        match : re.Match[str]
            A match of `CAPTION_PATTERN`.

        Returns
        -------
        str
            Figure label, e.g. `1` or `S1`.
        """

        prefix = "S" if match.group("supp") or match.group("prefix") else ""
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
            text_blocks = self._text_cache[page.number]  # type: ignore
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
                    i += 1
                    while i < num_blocks:
                        next_block = text_blocks[i]
                        # Blocks can include inner empty lines, so we should add lines by one.
                        capture_extended = False
                        lines_consumed = 0
                        for j in range(len(next_block.lines)):
                            next_line = next_block.lines[j]
                            if (
                                abs(
                                    next_line.rect.y1
                                    - block.lines[-1].rect.y1
                                    - self.line_pitch
                                )
                                < max(
                                    vertical_thr,
                                    self.LINE_PITCH_TOLERANCE * self.line_pitch,
                                )
                            ):
                                block = block + next_line
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
            return {page.number: tuple(fig_captures)}  # type: ignore

        captions: dict[int, tuple[FigureCaptionPDF, ...]] = {}
        for page_no in range(self.file.page_count):
            captions.update(_find_figure_captions_in_page(self.file[page_no]))
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
                and self._rect_key(clip_to_grid(block.rect, self.RECTS_CLIP_PRECISION))
                not in self._running_matter_rects
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

    def _is_side_caption(self, page_no: int, caption_rect: Rect) -> bool:

        page_captions = self._figure_captions_cache[page_no]
        page_caption = next(
            caption for caption in page_captions if caption.rect == caption_rect
        )
        return (
            caption_rect.width < self.paragraph_width.mean / 2
            and page_caption.lines_no > 1
        )

    def _get_figures_from_page(
        self,
        page_no: int,
    ) -> tuple[FigurePDF, ...]:
        """Internal method to get figure rectangles from a single page."""

        # Figure can cantain images, drawings and text.
        # This function determines bounding box of the figures on the page.
        # Algorithm:
        # 1. First, we determine maximum bounds of the figure as follows:
        #   1.1 Lower bound is upper bound of figure caption
        #   1.2 Upper bound is minimum value of lower bounds of:
        #       1.2.1 Text paragraph, which located above the figure caption
        #       1.2.2 Another figure caption, which is located above the figure caption
        #       1.2.3 Page header, which is located above the figure caption
        #   1.3 Left and right bounds should ba adjusted only if article layout is
        #   multicolumn.
        # Some figures have their captions located on the left or right side.
        # There is a heuristic to determine whether caption is side caption:
        # Width of caption is less than 50% of paragraph width and has maltiple lines.
        # There is another heuristic which can help to parse such figures:
        # If caption is side caption, then figure takes all width of the page
        # regardless of number of columns.

        result: list[FigurePDF] = []
        page = self.file[page_no]
        page_width = page.rect.width
        figure_captions = self._figure_captions_cache[page_no]

        # First we should find all figures with side captions
        side_capt_figures: list[FigurePDF] = []

        for caption in figure_captions:
            if self._is_side_caption(page_no, caption.rect):
                figure_rect = self._get_figure_rects_side_caption(
                    page_no=page_no, caption_rect=caption.rect
                )
                side_capt_figures.append(FigurePDF(figure_rect, caption))

        for caption in figure_captions:
            figure_rect = cast(Rect, copy(page.rect))
            figure_rect.x0, figure_rect.x1 = self._line_number_gutter(page_no)

            # skip side captions
            if self._is_side_caption(page_no, caption.rect):
                continue

            # Lower boundary
            figure_rect.y1 = caption.rect.y0

            # Upper boundary
            upper_bound_candidates: list[Rect] = []
            paragraphs_above = self.get_paragraph_rects(
                page_no=page_no, clip=figure_rect
            )

            # Columns could be shifted horizontaly so that the right boundary of
            # the left column can be located more to the right than half of page width
            # and vice versa
            min_x0_right_col = page_width * 0.4
            max_x1_left_col = page_width * 0.6
            # if we have two column layout we need to check if figure is
            # located in particular column and if so, check only paragraphs
            # in that column
            if self.columns_number > 1:
                # figure is in right column
                if caption.rect.x0 > min_x0_right_col:
                    paragraphs_above = [
                        rect for rect in paragraphs_above if rect.x0 > min_x0_right_col
                    ]
                # Figure is in left column
                if caption.rect.x1 < max_x1_left_col:
                    paragraphs_above = [
                        rect for rect in paragraphs_above if rect.x1 < max_x1_left_col
                    ]

            if paragraphs_above:
                lowest_paragraph = max(paragraphs_above, key=lambda x: x.y1)
                upper_bound_candidates.append(lowest_paragraph)

            # Search for figure caption above
            other_caption_rects = self.get_figure_caption_rects(
                page_no, clip=figure_rect
            )
            if self.columns_number > 1:
                # If figure is in right column, then we don't care
                # about other figures in left column
                if caption.rect.x0 > min_x0_right_col:
                    other_caption_rects = [
                        rect
                        for rect in other_caption_rects
                        if rect.x0 > min_x0_right_col
                    ]
                # Figure is in left column, then we don't care
                # about other figures
                elif caption.rect.x1 < max_x1_left_col:
                    other_caption_rects = [
                        rect
                        for rect in other_caption_rects
                        if rect.x1 < max_x1_left_col
                    ]

            if other_caption_rects:
                lowest_caption = max(other_caption_rects, key=lambda x: x.y1)
                upper_bound_candidates.append(lowest_caption)

            upper_bound_candidates.append(self.header_rect)
            upper_bound_candidates.extend(
                [
                    figure.rect
                    for figure in side_capt_figures
                    if figure.rect.y1 < caption.rect.y0
                ]
            )

            if upper_bound_candidates:
                lowest_bounding_rect = max(upper_bound_candidates, key=lambda x: x.y1)
                figure_rect.y0 = lowest_bounding_rect.y1

            # Side boundaries
            if self.columns_number > 1:
                if self.columns_number > 2:
                    warn(
                        f"Document {page.parent} page {page.number} has more than 2 columns. "
                        "Only 2 columns are supported for figure extraction."
                    )
                    result.extend(side_capt_figures)
                    return tuple(result)

                paragraph_rects = self.get_paragraph_rects(page_no, copy_rects=False)
                left_columns_x1 = [
                    rect.x1 for rect in paragraph_rects if rect.x1 < max_x1_left_col
                ]
                right_columns_x0 = [
                    rect.x0 for rect in paragraph_rects if rect.x0 > min_x0_right_col
                ]

                # Left boundary for figure in right column
                if caption.rect.x0 > min_x0_right_col:
                    # First try to set it slightly to the right from the most right
                    # position of the left column. But left column could be missing
                    if left_columns_x1:
                        figure_rect.x0 = max(left_columns_x1) + self.MARGIN
                    elif right_columns_x0:
                        figure_rect.x0 = max(right_columns_x0) - self.MARGIN

                # Right boundary for figure in left column
                if caption.rect.x1 < max_x1_left_col:
                    if right_columns_x0:
                        figure_rect.x1 = min(right_columns_x0) - self.MARGIN
                    elif left_columns_x1:
                        figure_rect.x1 = min(left_columns_x1) + self.MARGIN

            result.append(FigurePDF(figure_rect, caption))
        result.extend(side_capt_figures)
        for i in range(len(result)):
            result[i].rect = self._refine_figure_rect(page_no, result[i].rect)
        return tuple(result)

    def _refine_figure_rect(
        self,
        page_no: int,
        figure_rect: Rect,
    ) -> Rect:

        rects: list[Rect] = []
        rects.extend(self.get_text_rects(page_no, clip=figure_rect, copy_rects=False))
        rects.extend(
            self.get_drawing_rects(page_no, clip=figure_rect, copy_rects=False)
        )
        rects.extend(self.get_image_rects(page_no, clip=figure_rect, copy_rects=False))
        if not rects:
            return figure_rect
        refined_rect = copy(rects[0])
        for rect in rects:
            refined_rect.include_rect(rect)
        return refined_rect

    def _get_figure_rects_side_caption(
        self,
        page_no: int,
        caption_rect: Rect,
    ) -> Rect:
        """Internal method to get figure rect for side captions."""

        # The hardest task for such images is to determine upper and lower boundaries.
        # The upper boundary can be:
        # 1. Bottom of text paragraph above
        # 2. Page header
        # 3. Bottom of figure/table caption
        # 4. Bottom of another figure
        # The lower boundary can be:
        # 1. Top of text paragraph below
        # 2. Page footer
        # 3. Top of another figure
        #
        # The following heuristics is used: figure capture should be aligned to the top
        # or bottom of the image. Therefore we first check is there any objects above and below
        # the capture (all page wide is checked).

        # Determine initial side boundaries
        page = self.file[page_no]
        figure_rect = copy(page.rect)
        left_bound, right_bound = self._line_number_gutter(page_no)
        if caption_rect.x0 > page.rect.width / 2:
            figure_rect.x0 = max(self.MARGIN, left_bound)
            figure_rect.x1 = min(caption_rect.x0 - self.MARGIN, right_bound)
        else:
            figure_rect.x0 = max(caption_rect.x1 + self.MARGIN, left_bound)
            figure_rect.x1 = min(page.rect.width - self.MARGIN, right_bound)

        # Check if figure caption is aligned at bottom of the figure
        test_rect = Rect(
            self.MARGIN,
            caption_rect.y1 + self.MARGIN,
            page.rect.width - self.MARGIN,
            caption_rect.y1 + 2 * self.MARGIN,
        )
        text_rects = self.get_text_rects(page_no)
        drawing_rects = self.get_drawing_rects(page_no)
        image_rects = self.get_image_rects(page_no)
        has_no_objects = True
        for obj in chain(text_rects, drawing_rects, image_rects):
            if test_rect.intersects(obj):
                has_no_objects = False
                break

        caption_rects = self.get_figure_caption_rects(page_no)
        paragraph_rects = self.get_paragraph_rects(page_no)
        # If we did not found any objects below, then capture is bottom-aligned
        if has_no_objects:
            figure_rect.y1 = caption_rect.y1 + self.MARGIN

            upper_bound_candidates = [
                rect
                for rect in chain(caption_rects, paragraph_rects, [self.header_rect])
                if rect and rect.y1 < caption_rect.y0
            ]
            if upper_bound_candidates:
                figure_rect.y0 = (
                    max(upper_bound_candidates, key=lambda x: x.y1).y1 + self.MARGIN
                )
        # Otherwise caption is top aligned
        else:
            figure_rect.y0 = caption_rect.y0 - self.MARGIN
            lower_bound_candidates = [
                rect
                for rect in chain(caption_rects, paragraph_rects)
                if rect and rect.y0 > caption_rect.y1
            ]
            if lower_bound_candidates:
                figure_rect.y1 = (
                    min(lower_bound_candidates, key=lambda x: x.y0).y0 - self.MARGIN
                )

        return figure_rect
