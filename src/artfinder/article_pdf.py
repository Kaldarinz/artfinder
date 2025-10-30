"""
Article class for PDF analysis and figure extraction.

This module provides a comprehensive class for analyzing scientific articles in PDF format,
extracting figures, captions, and analyzing document structure.
"""

import re
from collections import namedtuple, Counter
from dataclasses import dataclass, field
from copy import deepcopy, copy
from functools import cached_property, lru_cache
from itertools import chain
from math import floor
import os
from os import PathLike
from pathlib import Path
from typing import Optional, cast, Any, List, Sequence, Union, Mapping, Iterable
from pprint import pprint
from warnings import warn

import pandas as pd
import pymupdf
from pymupdf import Document, Page, Rect, Matrix, Pixmap, Shape
from sklearn.cluster import DBSCAN

from artfinder.dataclasses import (
    Color,
    DrawingObjectPDF,
    ImageInfoPDF,
    TextBlockPDF,
    KeyedDict,
    Size,
    FigureCaptionPDF,
    DocumentElementsPDF,
    FigurePDF,
)
from artfinder.helpers import (
    clip_to_grid,
    rects_equal
)

class ArticlePDF:
    """
    A class for analyzing scientific articles in PDF format.

    This class provides methods for extracting figures, captions, analyzing document
    structure (columns, paragraphs), and identifying document headers.

    Figures are composed of vector graphics (drawings) and raster images and text elements.
    """

    COLUMN_NO_FITTING_FACTOR = 1.1
    "Factor to adjust column fitting when estimating number of columns."
    HEADER_MAX_FRACTION = 0.1
    "Maximum fraction of page height to consider for header detection."
    RECTS_EQUAL_THR = 0.1
    "Coordinate threshold for considering two rectangles equal."
    RECTS_CLIP_PRECISION = 0
    "Precidion digits for clipping rects coordinates."
    MAX_IMAGE_AREA = 0.8
    "Maximum fraction of page area for an image to be considered valid."
    MARGIN = 2
    "Margin in points for rectangles."

    def __init__(self, pdf_path: PathLike | str, identifier: str = ""):
        """
        Initialize Article with a PDF file.

        Parameters
        ----------
        pdf_path : PathLike | str
            Path to the PDF file to analyze.

        Raises
        ------
        FileNotFoundError
            If the PDF file does not exist.
        ValueError
            If the file cannot be opened as a PDF.
        """
        self.identifier = identifier
        self.path = Path(pdf_path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            self.file = pymupdf.open(str(self.path))
            if len(self.identifier) == 0:
                self.identifier = self.file.name.split("/")[-1] # type: ignore
        except Exception as e:
            raise ValueError(f"Failed to open PDF file: {e}")

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
        self._figures_cache: dict[int, tuple[FigurePDF, ...]] = KeyedDict(
            self._get_figures_from_page
        )
        """Figures keyed by page number."""
        self._figure_no_to_page_ind: dict[int, int] = {}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the PDF."""
        self.close()

    def close(self):
        """Close the PDF document."""
        if hasattr(self, "pdf") and self.file:
            self.file.close()

    def __repr__(self):
        return f"Article('{self.path.name}', pages={len(self.file)})"

    # Cached properties for expensive computations

    @cached_property
    def paragraph_width(self) -> Size:
        """
        Paragraph width for the entire document.

        Returns
        -------
        Width
            Named tuple with mean, min, max of paragraph widths.
        """

        return self._calc_paragraph_width()

    @cached_property
    def columns_number(self) -> int:
        """
        Number of columns in the document.

        Returns
        -------
        int
            The estimated number of columns (1, 2, etc.).
        """
        all_widths = [page.rect.width for page in self.file]
        if len(all_widths):
            page_width = sum(all_widths) / len(all_widths)
        else:
            raise ValueError("No pages with valid width.")
        return floor(
            page_width / (self.paragraph_width.mean * self.COLUMN_NO_FITTING_FACTOR)
        )

    @cached_property
    def header_rect(self) -> Rect:
        """
        Document header rectangle.

        Returns
        -------
        Rect | None
            The header rectangle if found, otherwise None.
        """

        # We assume that all pages in the pdf file have the same size
        search_rect = cast(Rect, self.file[0].rect)

        search_height = search_rect.height * self.HEADER_MAX_FRACTION
        search_rect.y1 = search_height

        cnt = Counter()
        header_found = False
        clip_precision = self.RECTS_CLIP_PRECISION
        while not header_found:
            text_rects = self.get_text_rects(clip=search_rect)
            text_rects_clipped = [
                clip_to_grid(rect, clip_precision) for rect in text_rects
            ]
            cnt.update(text_rects_clipped)

            drawing_rects = self.get_drawing_rects(clip=search_rect)
            drawing_rects_clipped = [
                clip_to_grid(rect, clip_precision) for rect in drawing_rects
            ]
            cnt.update(drawing_rects_clipped)

            image_rects = self.get_image_rects(clip=search_rect)
            image_rects_clipped = [
                clip_to_grid(rect, clip_precision) for rect in image_rects
            ]
            cnt.update(image_rects_clipped)
            most_common_cnts = set(cnt.values()).difference([1])
            if len(most_common_cnts):
                header_found = True
            else:
                clip_precision -= 1
                if clip_precision < -1:
                    warn(f"Header rectangle not found for {self}.")
                    self._header_rect = Rect()
                    return self._header_rect
        # Use only 3 most popular values for rect counts
        most_common_cnts = sorted(most_common_cnts, reverse=True)[:1]
        most_common_rects = {k: v for k, v in cnt.items() if v in most_common_cnts}
        lowest_bound = max(most_common_rects.keys(), key=lambda x: x.y1).y1
        search_rect.y1 = lowest_bound
        return search_rect

    @cached_property
    def figure_captions_cache(self) -> dict[int, tuple[FigureCaptionPDF, ...]]:
        """Figure captions in the document."""

        return self._find_figure_captions()

    @cached_property
    def figures(self) -> dict[int, FigurePDF]:
        return self.get_figures()

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
        if page_no is None:
            blocks = chain.from_iterable(
                [self._text_cache[i] for i in range(self.file.page_count)]
            )
        else:
            blocks = self._text_cache[page_no]
        for block in blocks:
            if clip is None or clip.contains(block.rect):
                if min_len is None or len(block.text.strip()) >= min_len:
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
            rect_type="figure_captions",
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

        text_rects = self.get_text_rects(
            page_no=page_no, clip=clip, copy_rects=copy_rects
        )
        paragraph_rects = [
            rect
            for rect in text_rects
            if rect.width >= self.paragraph_width.min
            and rect.width <= self.paragraph_width.max
        ]
        return paragraph_rects

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
    ) -> dict[int, FigurePDF]:
        """
        Get figures from the document.

        Parameters
        ----------
        page_no : Page | int | None
            Page number (0-indexed). If None, return figures for all document
        clip : Rect | None, optional
            Rectangle to clip the figure blocks.
        copy_rects : bool, default=True
            Whether to return copies of the rectangles or references.

        Returns
        -------
        dict[int, Figure]
            Dictionary of figures keyed by figure number.
        """

        if isinstance(page_no, Page):
            page_no = cast(int, page_no.number)

        if page_no is None:
            pages = range(self.file.page_count)
        else:
            pages = [page_no]

        result: dict[int, FigurePDF] = {}
        for page in pages:
            figures = self._figures_cache[page]
            for figure in figures:
                if clip is not None and not clip.contains(figure.rect):
                    continue
                match = re.search(r"\d+", figure.caption.matched_pattern)
                if match is None:
                    warn(
                        "Could not extract figure number from caption pattern "
                        + f"'{figure.caption.matched_pattern}' in {self}"
                    )
                    continue
                fig_no = int(match.group())
                # Check for duplicates
                if fig_no in result:
                    warn(
                        f"Duplicate figure number {fig_no} found on page {page} in {self}. Overwriting."
                    )

                self._figure_no_to_page_ind[fig_no] = page
                if copy_figures:
                    result[fig_no] = deepcopy(figure)
                else:
                    result[fig_no] = figure

        return result

    def get_figure_drawings(
        self,
        figure_no: int | None = None,
        page_index: int | None = None,
        copy_objects: bool = False,
    ) -> list[DrawingObjectPDF]:
        """
        Get drawing objects for a specific figure.

        Parameters
        ----------
        figure_no : int | None, optional
            Figure number to get drawings for. If None, gets drawings for all figures.
        page_index : int | None, optional
            Page index to get figures from. If None, get all figures specified by figure_no.
        copy_objects : bool, default=False
            Whether to return copies of the drawing objects or references.

        Returns
        -------
        list[DrawingObject]
            List of drawing objects for the specified figure(s).
        """

        return self._get_figure_component_by_type(
            figure_no=figure_no,
            page_index=page_index,
            component_type="drawings",
            copy_objects=copy_objects
        )

    def get_figure_images(
        self,
        figure_no: int | None = None,
        page_index: int | None = None,
        copy_objects: bool = False,
    ) -> list[ImageInfoPDF]:
        """
        Get image objects for a specific figure.

        Parameters
        ----------
        figure_no : int | None, optional
            Figure number to get images for. If None, gets images for all figures.
        page_index : int | None, optional
            Page index to get figures from. If None, get all figures specified by figure_no.
        copy_objects : bool, default=False
            Whether to return copies of the drawing objects or references.

        Returns
        -------
        list[DrawingObject]
            List of drawing objects for the specified figure(s).
        """

        return self._get_figure_component_by_type(
            figure_no=figure_no,
            page_index=page_index,
            component_type="images",
            copy_objects=copy_objects
        )

    def get_figure_texts(
        self,
        figure_no: int | None = None,
        page_index: int | None = None,
        copy_objects: bool = False,
    ) -> list[TextBlockPDF]:
        """
        Get text, associated with a specific figure.

        Parameters
        ----------
        figure_no : int | None, optional
            Figure number to get drawings for. If None, gets drawings for all figures.
        page_index : int | None, optional
            Page index to get figures from. If None, get all figures specified by figure_no.
        copy_objects : bool, default=False
            Whether to return copies of the drawing objects or references.

        Returns
        -------
        list[TextBlock]
            List of text blocks for the specified figure(s).
        """

        return self._get_figure_component_by_type(
            figure_no=figure_no,
            page_index=page_index,
            component_type="text",
            copy_objects=copy_objects
        )

    def extract_figure_drawings(
        self,
        figure_no: int | None = None,
        page_index: int | None = None,
        highlight_white: bool = True,
        output_path: PathLike | str = Path("figures"),
        dpi: int = 150,
    ) -> list[Path]:
        """
        Extract vector graphics component of a figure as a separate image.

        Parameters
        ----------
        figure_no : int | None, optional
            Figure number to extract. If None, extracts all figures.
        page_index : int | None, optional
            Page index to extract figures from. If None, extracts all figures
            specified by figure_no.
        highlight_white : bool, default=True
            Whether to highlight white drawings by placing a border around them.
        output_path : PathLike | str, default=Path("figures")
            Path to save the extracted figure images.
        dpi : int, default=150
            DPI for the output images.

        Returns
        -------
        list[Path]
            Paths to the saved figure images.
        """

        output_path = Path(output_path)

        figures = self.get_figures(page_index)
        if figure_no is not None:
            figures = {figure_no: figures[figure_no]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        paths: list[Path] = []
        output_path.mkdir(parents=True, exist_ok=True)
        for fig_no in figures:
            fig_drawings = self.get_figure_drawings(
                fig_no, copy_objects=highlight_white
            )
            if fig_drawings:
                if highlight_white:
                    for drawing in fig_drawings:
                        if drawing.fill == (1.0, 1.0, 1.0):
                            drawing.width = 1
                drawings_page = self._make_drawings(fig_drawings)
                pixmap = drawings_page.get_pixmap(dpi=dpi, clip=figures[fig_no].rect)

                path = output_path / Path(f"{self.identifier}_fig_{fig_no}_vectors.png")
                pixmap.save(path, output="PNG")
                paths.append(path)

        return paths

    def extract_figure_text(
        self,
        figure_no: int | None = None,
        page_index: int | None = None,
        output_path: PathLike | str = Path("figures"),
        dpi: int = 150,
    ) -> list[Path]:
        """
        Extract text component of a figure as a separate image.

        Parameters
        ----------
        figure_no : int | None, optional
            Figure number to extract. If None, extracts all figures.
        page_index : int | None, optional
            Page index to extract figures from. If None, extracts all figures
            specified by figure_no.
        output_path : PathLike | str, default=Path("figures")
            Path to save the extracted figure images.
        dpi : int, default=150
            DPI for the output images.

        Returns
        -------
        list[Path]
            Paths to the saved figure images.
        """

        output_path = Path(output_path)

        figures = self.get_figures(page_index)
        if figure_no is not None:
            figures = {figure_no: figures[figure_no]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        paths: list[Path] = []
        output_path.mkdir(parents=True, exist_ok=True)
        for fig_no in figures:
            figure_texts = self.get_figure_texts(fig_no)
            if figure_texts:
                figure_text_page = self._make_text(figure_texts)
                pixmap = figure_text_page.get_pixmap(dpi=dpi, clip=figures[fig_no].rect)

                path = output_path / Path(f"{self.identifier}_fig_{fig_no}_text.png")
                pixmap.save(path, output="PNG")
                paths.append(path)

        return paths

    def extract_figure_images(
        self,
        figure_no: int | None = None,
        page_index: int | None = None,
        output_path: PathLike | str = Path("figures"),
        dpi: int = 150,
    ) -> list[Path]:
        """
        Extract images component of a figure as a separate image.

        Parameters
        ----------
        figure_no : int | None, optional
            Figure number to extract. If None, extracts all figures.
        page_index : int | None, optional
            Page index to extract figures from. If None, extracts all figures
            specified by figure_no.
        output_path : PathLike | str, default=Path("figures")
            Path to save the extracted figure images.
        dpi : int, default=150
            DPI for the output images.

        Returns
        -------
        list[Path]
            Paths to the saved figure images.
        """

        output_path = Path(output_path)

        figures = self.get_figures(page_index)
        if figure_no is not None:
            figures = {figure_no: figures[figure_no]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        paths: list[Path] = []
        output_path.mkdir(parents=True, exist_ok=True)
        for fig_no in figures:
            figure_images = self.get_figure_images(fig_no)
            if figure_images:
                figure_image_page = self._make_image(figure_images)
                pixmap = figure_image_page.get_pixmap(dpi=dpi, clip=figures[fig_no].rect)

                path = output_path / Path(f"{self.identifier}_fig_{fig_no}_image.png")
                pixmap.save(path, output="PNG")
                paths.append(path)

        return paths

    def extract_figures(
        self,
        figure_no: int | None = None,
        page_index: int | None = None,
        output_path: PathLike | str = Path("figures"),
        extract_bases: bool = False,
        dpi: int = 150,
    ) -> list[Path]:
        """
        Extract a specific figure as a separate images.

        Parameters
        ----------
        figure_no : int | None, optional
            Figure number to extract. If None, extracts all figures.
        page_index : int | None, optional
            Page index to extract figures from. If None, extracts all figures
            specified by figure_no.
        output_path : PathLike | str, default=Path("figures")
            Path to save the extracted figure images.
        extract_bases : bool, default=False
            Whether to also extract the base components (drawings, images, text)
        dpi : int, default=150
            DPI for the output images.

        Returns
        -------
        list[Path]
            Paths to the saved figure images.
        """

        output_path = Path(output_path)

        figures = self.get_figures(page_index)
        if figure_no is not None:
            figures = {figure_no: figures[figure_no]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        paths = []
        output_path.mkdir(parents=True, exist_ok=True)
        for fig_no in figures:
            page_ind = self._figure_no_to_page_ind[fig_no]
            pixmap = self.file[page_ind].get_pixmap(dpi=dpi, clip=figures[fig_no].rect)

            path = output_path / Path(f"{self.identifier}_fig_{fig_no}.png")
            pixmap.save(path, output="PNG")
            paths.append(path)
            if extract_bases:
                self.extract_figure_drawings(
                    figure_no=fig_no,
                    page_index=page_ind,
                    output_path=output_path,
                    dpi=dpi
                )
                self.extract_figure_images(
                    figure_no=fig_no,
                    page_index=page_ind,
                    output_path=output_path,
                    dpi=dpi
                )
                self.extract_figure_text(
                    figure_no=fig_no,
                    page_index=page_ind,
                    output_path=output_path,
                    dpi=dpi
                )

        return paths

    def mark(
        self,
        elements: list[str] | str | None,
        page_number: int | None = None,
        output_path: Optional[PathLike | str] = None,
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
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.FIGURE_CAPTION]:
                    rects.extend(
                        self.get_figure_caption_rects(page.number, copy_rects=False)
                    )
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.PARAGRAPH]:
                    rects.extend(
                        self.get_paragraph_rects(page.number, copy_rects=False)
                    )
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.HEADER]:
                    rects.append(self.header_rect)
                if element in [DocumentElementsPDF.ALL, DocumentElementsPDF.FIGURE]:
                    rects.extend(self.get_figure_rects(page.number, copy_rects=False))

            for rect in rects:
                shape = page.new_shape()
                shape.draw_rect(rect)
                shape.finish(color=stroke_color, width=stroke_width)
                shape.commit()

        if output_path is None:
            output_path = Path(f"marked_{self.path.name}").resolve()
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
        figure_no: int | None = None,
        page_index: int | None = None,
        copy_objects: bool = False,
    ) -> list:
        """
        Internal method to get figure components (images, drawings, texts).
        """

        figures = self.get_figures(page_index)
        if figure_no is not None:
            figures = {figure_no: figures[figure_no]}

        if len(figures) == 0:
            warn("No figures found to extract.")
            return []

        components = []
        for fig_no in figures:
            page_ind = self._figure_no_to_page_ind[fig_no]
            page_components = getattr(self, f"_{component_type}_cache")[page_ind]

            components.extend(
                (
                    deepcopy(component) if copy_objects else component
                    for component in page_components
                    if figures[fig_no].rect.contains(component.rect)
                )
            )
        return components

    def _get_text_blocks_from_page(self, page_no: int) -> tuple[TextBlockPDF, ...]:
        """
        Internal method to get all non blank text blocks from a page.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).
        """

        result: list[TextBlockPDF] = []
        text_blocks = [
            TextBlockPDF.from_dict(block) # type: ignore
            for block in self.file[page_no].get_text(option="dict")["blocks"]  # type: ignore
            if block["type"] == 0 # type: ignore
        ]
        for block in text_blocks:
            # Skip blank blocks
            if len(block.text) == 0 or block.text.isspace():
                continue
            result.append(block)
        return tuple(result)

    def _get_drawing_objs_from_page(self, page_no: int) -> tuple[DrawingObjectPDF, ...]:
        """
        Internal method to get all drawing objects from a page.

        Parameters
        ----------
        page_no : int
            Page number (0-indexed).
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
            (
                image
                for image in all_images
                if abs(image.rect) < page_area * self.MAX_IMAGE_AREA
            )
        )

    def _make_drawings(
        self, drawings: Iterable[DrawingObjectPDF], page: Page | None = None
    ) -> Page:
        """
        Internal method to render drawing objects.

        Parameters
        ----------
        drawings : Iterable[DrawingObject]
            Drawing objects to render.
        page : Page | None
            Page to render drawings on. If None, creates a new blank page.

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
        text_kwargs : Iterable[dict]
            Text objects to render.
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
        images : Iterable[PdfImageInfo]
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

        Parameters
        ----------
        eps : float
            The maximum distance between two samples for one to be considered as
            in the neighborhood of the other.
        """

        text_blocks = pd.DataFrame(
            [[rect.x0, rect.y0, rect.x1, rect.y1] for rect in self.get_text_rects()],
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

    def _find_figure_captions(self) -> dict[int, tuple[FigureCaptionPDF, ...]]:
        """Internal method to find figure captions on page."""

        def _find_figure_captions_in_page(
            page: Page,
        ) -> dict[int, tuple[FigureCaptionPDF]]:
            # Pattern: optional whitespace, 'Fig', optional '.' or 'ure' or 'ure.',
            # whitespace, digits, optional trailing '.'
            pattern = r"^\s*Fig(?:\.|ure\.?|)\s+\d+\.?"
            vertical_thr = 1
            fig_captures = []
            text_blocks = self._text_cache[page.number]  # type: ignore
            num_blocks = len(text_blocks)
            i = 0
            while i < num_blocks:
                block = text_blocks[i]
                text = block.text.strip()
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    # Some paragraphs can also begin with `pattern`, but in almost all cases
                    # font of `pattern` in figure caption is different from font of paragraph.
                    # Therefore we save font properties for the mathed block for later use.
                    font_props = (
                        block.lines[0].spans[0].char_flags,
                        block.lines[0].spans[0].flags,
                        block.lines[0].spans[0].font
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
                        if (
                            abs(next_block.lines[0].rect.y1 - block.lines[-1].rect.y1 - block.lines[-1].rect.height)
                            < vertical_thr
                        ):
                            block = block + next_block
                            i += 1 
                            continue
                        break
                    # remove matched pattern and extra blank lines from text
                    text = block.text.replace(pattern_text, "").strip()
                    fig_captures.append(
                        FigureCaptionPDF(
                            **{
                                "matched_pattern": pattern_text,
                                "font_props": font_props,
                                "text": text,
                                "x0": block.rect.x0,
                                "y0": block.rect.y0,
                                "x1": block.rect.x1,
                                "y1": block.rect.y1,
                                "rect": copy(block.rect),
                            }
                        )
                    )
                i += 1
            return {page.number: tuple(fig_captures)}  # type: ignore

        captions: dict[int, tuple[FigureCaptionPDF]] = {}
        for page in self.file:
            captions.update(_find_figure_captions_in_page(page))
        # Some text paragraphs can start the same way as figure caption.
        # Text paragraph usually have their font_flag different from font_flag
        # of matched_pattern in real figure caption.
        # Therefore we only use found FigureCaption with the most common font_flag
        most_common_flags = Counter(
            (
                caption.font_props
                for page_captions in captions.values()
                for caption in page_captions
            )
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

    def _is_side_caption(self, page_no: int, caption_rect: Rect) -> bool:

        page_captions = self.figure_captions_cache[page_no]
        page_caption = [
            caption for caption in page_captions if caption.rect == caption_rect
        ][0]
        lines_no = len(page_caption.text.splitlines())
        if caption_rect.width < self.paragraph_width.mean / 2 and lines_no > 1:
            return True
        return False

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
        figure_captions = self.figure_captions_cache[page_no]

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
            figure_captions = self.get_figure_caption_rects(page_no, clip=figure_rect)
            if self.columns_number > 1:
                # If figure is in right column, then we don't care
                # about other figures in left column
                if caption.rect.x0 > min_x0_right_col:
                    figure_captions = [
                        rect for rect in figure_captions if rect.x0 > min_x0_right_col
                    ]
                # Figure is in left column, then we don't care
                # about other figures
                elif caption.rect.x1 < max_x1_left_col:
                    figure_captions = [
                        rect for rect in figure_captions if rect.x1 < max_x1_left_col
                    ]

            if figure_captions:
                lowest_caption = max(figure_captions, key=lambda x: x.y1)
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
                    else:
                        figure_rect.x0 = max(right_columns_x0) - self.MARGIN

                # Right boundary for figure in right column
                if caption.rect.x1 < max_x1_left_col:
                    if right_columns_x0:
                        figure_rect.x1 = min(right_columns_x0) - self.MARGIN
                    else:
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
        if caption_rect.x0 > page.rect.width / 2:
            figure_rect.x0 = self.MARGIN
            figure_rect.x1 = caption_rect.x0 - self.MARGIN
        else:
            figure_rect.x0 = caption_rect.x1 + self.MARGIN
            figure_rect.x1 = page.rect.width - self.MARGIN

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

    def _extract_figure_no(self, figure: FigurePDF) -> int:

        match = re.search(r"\d+", figure.caption.matched_pattern)
        if match is None:
            warn(
                "Could not extract figure number from caption pattern "
                + f"'{figure.caption.matched_pattern}' in {self}"
            )
            return 0
        return int(match.group())



