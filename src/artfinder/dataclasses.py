# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT

from typing import NamedTuple
from enum import StrEnum
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union
from pymupdf import Rect, Matrix



class CrossrefRateLimit(NamedTuple):
    """
    A class to represent the Crossref rate limit.
    """

    limit: int
    interval: int

class CrossrefResource(StrEnum):
    """
    A class to represent the Crossref API endpoints.
    """

    WORKS = 'works'
    FUNDERS = 'funders'
    MEMBERS = 'members'
    PREFIXES = 'prefixes'
    TYPES = 'types'
    JOURNALS = 'journals'


class CrossrefQueryField(StrEnum):
    """
    Crossref query fields.
    """

    AFFILIATION = 'affiliation'
    AUTHOR = 'author'
    BIBLIOGRAPHIC = 'bibliographic'
    CHAIR = 'chair'
    CONTAINER_TITLE = 'container-title'
    CONTRIBUTOR = 'contributor'
    EDITOR = 'editor'
    EVENT_ACRONYM = 'event-acronym'
    EVENT_LOCATION = 'event-location'
    EVENT_NAME = 'event-name'
    EVENT_SPONSOR = 'event-sponsor'
    EVENT_THEME = 'event-theme'
    FUNDER = 'funder-name'
    PUBLICHER = 'publisher-name'
    PUBLISHER_LOCATION = 'publisher-location'
    TRANSLATOR = 'translator'

class CrossrefSelectField(StrEnum):
    """
    Valid Crossref filter fields.
    """

    DOI = 'DOI'
    ISBN = 'ISBN'
    ISSN = 'ISSN'
    URL = 'URL'
    ABSTRACT = 'abstract'
    ACCEPTED = 'accepted'
    ALTERNATIVE_ID = 'alternative-id'
    APPROVED = 'approved'
    ARCHIVE = 'archive'
    ARTICLE_NUMBER = 'article-number'
    ASSERTION = 'assertion'
    AUTHOR = 'author'
    CHAIR = 'chair'
    CLINICAL_TRIAL_NUMBER = 'clinical-trial-number'
    CONTAINER_TITLE = 'container-title'
    CONTENT_CREATED = 'content-created'
    CONTENT_DOMAIN = 'content-domain'
    CREATED = 'created'
    DEGREE = 'degree'
    DEPOSITED = 'deposited'
    EDITOR = 'editor'
    EVENT = 'event'
    FUNDER = 'funder'
    GROUP_TITLE = 'group-title'
    INDEXED = 'indexed'
    IS_REFERENCED_BY_COUNT = 'is-referenced-by-count'
    ISSN_TYPE = 'issn-type'
    ISSUE = 'issue'
    ISSUED = 'issued'
    LICENSE = 'license'
    LINK = 'link'
    MEMBER = 'member'
    ORIGINAL_TITLE = 'original-title'
    PAGE = 'page'
    POSTED = 'posted'
    PREFIX = 'prefix'
    PUBLISHED = 'published'
    PUBLISHED_ONLINE = 'published-online'
    PUBLISHED_PRINT = 'published-print'
    PUBLISHER = 'publisher'
    PUBLISHER_LOCATION = 'publisher-location'
    REFERENCE = 'reference'
    REFERENCES_COUNT = 'references-count'
    RELATION = 'relation'
    SCORE = 'score'
    SHORT_CONTAINER_TITLE = 'short-container-title'
    SHORT_TITLE = 'short-title'
    STANDARDS_BODY = 'standards-body'
    SUBJECT = 'subject'
    SUBTITLE = 'subtitle'
    TITLE = 'title'
    TYPE = 'type'
    TRANSLATOR = 'translator'
    UPDATE_POLICY = 'update-policy'
    UPDATE_TO = 'update-to'
    UPDATED_BY = 'updated-by'
    VOLUME = 'volume'

class CrossrefFilterField(StrEnum):
    """
    Enum for Crossref filter fields.
    """
    # Members generated from VALIDATORS keys
    ALTERNATIVE_ID = "alternative-id"
    ARCHIVE = "archive"
    ARTICLE_NUMBER = "article-number"
    ASSERTION = "assertion"
    ASSERTION_GROUP = "assertion-group"
    AWARD_FUNDER = "award.funder"
    AWARD_NUMBER = "award.number"
    CATEGORY_NAME = "category-name"
    CLINICAL_TRIAL_NUMBER = "clinical-trial-number"
    CONTAINER_TITLE = "container-title"
    CONTENT_DOMAIN = "content-domain"
    DIRECTORY = "directory"
    DOI = "doi"
    FROM_ACCEPTED_DATE = "from-accepted-date"
    FROM_CREATED_DATE = "from-created-date"
    FROM_DEPOSIT_DATE = "from-deposit-date"
    FROM_EVENT_END_DATE = "from-event-end-date"
    FROM_EVENT_START_DATE = "from-event-start-date"
    FROM_INDEX_DATE = "from-index-date"
    FROM_ISSUED_DATE = "from-issued-date"
    FROM_ONLINE_PUB_DATE = "from-online-pub-date"
    FROM_POSTED_DATE = "from-posted-date"
    FROM_PRINT_PUB_DATE = "from-print-pub-date"
    FROM_PUB_DATE = "from-pub-date"
    FROM_UPDATE_DATE = "from-update-date"
    FULL_TEXT_APPLICATION = "full-text.application"
    FULL_TEXT_TYPE = "full-text.type"
    FULL_TEXT_VERSION = "full-text.version"
    FUNDER = "funder"
    FUNDER_DOI_ASSERTED_BY = "funder-doi-asserted-by"
    GROUP_TITLE = "group-title"
    HAS_ABSTRACT = "has-abstract"
    HAS_AFFILIATION = "has-affiliation"
    HAS_ARCHIVE = "has-archive"
    HAS_ASSERTION = "has-assertion"
    HAS_AUTHENTICATED_ORCID = "has-authenticated-orcid"
    HAS_AWARD = "has-award"
    HAS_CLINICAL_TRIAL_NUMBER = "has-clinical-trial-number"
    HAS_CONTENT_DOMAIN = "has-content-domain"
    HAS_DOMAIN_RESTRICTION = "has-domain-restriction"
    HAS_EVENT = "has-event"
    HAS_FULL_TEXT = "has-full-text"
    HAS_FUNDER = "has-funder"
    HAS_FUNDER_DOI = "has-funder-doi"
    HAS_LICENSE = "has-license"
    HAS_ORCID = "has-orcid"
    HAS_REFERENCES = "has-references"
    HAS_RELATION = "has-relation"
    HAS_UPDATE = "has-update"
    HAS_UPDATE_POLICY = "has-update-policy"
    IS_UPDATE = "is-update"
    ISBN = "isbn"
    ISSN = "issn"
    LICENSE_DELAY = "license.delay"
    LICENSE_URL = "license.url"
    LICENSE_VERSION = "license.version"
    LOCATION = "location"
    MEMBER = "member"
    ORCID = "orcid"
    PREFIX = "prefix"
    RELATION_OBJECT = "relation.object"
    RELATION_OBJECT_TYPE = "relation.object-type"
    RELATION_TYPE = "relation.type"
    TYPE = "type"
    TYPE_NAME = "type-name"
    UNTIL_ACCEPTED_DATE = "until-accepted-date"
    UNTIL_CREATED_DATE = "until-created-date"
    UNTIL_DEPOSIT_DATE = "until-deposit-date"
    UNTIL_EVENT_END_DATE = "until-event-end-date"
    UNTIL_EVENT_START_DATE = "until-event-start-date"
    UNTIL_INDEX_DATE = "until-index-date"
    UNTIL_ISSUED_DATE = "until-issued-date"
    UNTIL_ONLINE_PUB_DATE = "until-online-pub-date"
    UNTIL_POSTED_DATE = "until-posted-date"
    UNTIL_PRINT_PUB_DATE = "until-print-pub-date"
    UNTIL_PUB_DATE = "until-pub-date"
    UNTIL_UPDATE_DATE = "until-update-date"
    UPDATE_TYPE = "update-type"
    UPDATES = "updates"


class DocumentType(StrEnum):
    """Valid document types."""

    BOOK = 'book'
    BOOK_CHAPTER = 'book-chapter'
    BOOK_SET = 'book-set'
    BOOK_SERIES = 'book-series'
    BOOK_PART = 'book-part'
    BOOK_SECTION = 'book-section'
    BOOK_TRACK = 'book-track'
    REFERENCE_BOOK = 'reference-book'
    EDITED_BOOK = 'edited-book'
    MONOGRAPH = 'monograph'
    REPORT = 'report'
    PROCEEDINGS = 'proceedings'
    PROCEEDINGS_ARTICLE = 'proceedings-article'
    JOURNAL = 'journal'
    JOURNAL_ARTICLE = 'journal-article'
    JOURNAL_VOLUME = 'journal-volume'
    JOURNAL_ISSUE = 'journal-issue'
    OTHER = 'other'
    REFERENCE_ENTRY = 'reference-entry'
    COMPONENT = 'component'
    REPORT_SERIES = 'report-series'
    STANDARD = 'standard'
    STANDARD_SERIES = 'standard-series'
    POSTER_CONTENT = 'poster-content'
    DISSERTATION = 'dissertation'
    DATASET = 'dataset'

#############################
### ArticlePDF containers ###
#############################
class DocumentElementsPDF(StrEnum):
    FIGURE = "figures"
    FIGURE_CAPTION = "figure_captions"
    TEXT = "text"
    PARAGRAPH = "paragraphs"
    IMAGE = "images"
    DRAWING = "drawings"
    HEADER = "headers"
    ALL = "all"

@dataclass(frozen=True)
class Color:
    values: Union[Sequence[float], List[float]]

    def __post_init__(self):
        if len(self.values) not in (1, 3, 4):
            raise ValueError("Color must have 1 (GRAY), 3 (RGB), or 4 (CMYK) values.")
        if not all(0.0 <= v <= 1.0 for v in self.values):
            raise ValueError("Color values must be floats between 0 and 1.")

    def __repr__(self):
        if len(self.values) == 1:
            return f"GRAY({self.values[0]:.2f})"
        elif len(self.values) == 3:
            return f"RGB{tuple(round(v,2) for v in self.values)}"
        else:
            return f"CMYK{tuple(round(v,2) for v in self.values)}"


@dataclass
class DrawingObjectPDF:
    """
    List of this objects is returned by page.get_drawing() method.

    Default values correspond to default values of shape.finish() method.
    """

    closePath: bool = True
    """
    Causes the end point of a drawing to be automatically
    connected with the starting point (by a straight line)
    """
    color: tuple[int, ...] | None = field(default_factory=lambda: (0,))
    """
    Stroke color can be specified as tuples or list of of floats
    from 0 to 1. These sequences must have a length of 1 (GRAY), 3 (RGB) or
    4 (CMYK). For GRAY colorspace, a single float instead of the unwieldy
    (float,) or [float] is also accepted. Accept (default) or use None to
    not use the parameter.
    """
    dashes: Optional[str] = None
    """Causes lines to be drawn dashed. The general format is "[n m] p" of (up to)
    3 floats denoting pixel lengths. n is the dash length, m (optional) is the
    subsequent gap length, and p (the “phase” - required, even if 0!) specifies how
    many pixels should be skipped before the dashing starts. If m is omitted, it defaults to n.
    A continuous line (no dashes) is drawn with "[] 0" or None or "". 
    
    Examples:
    * Specifying "[3 4] 0" means dashes of 3 and gaps of 4 pixels following each other.
    * "[3 3] 0" and "[3] 0" do the same thing."""
    even_odd: bool = False
    """
    request the “even-odd rule” for filling operations. Default is False, so
    that the “nonzero winding number rule” is used. These rules are alternative
    methods to apply the fill color where areas overlap. Only with fairly
    complex shapes a different behavior is to be expected with these rules.
    """
    fill: tuple[int, ...] | None = None
    """
    Fill color can be specified as tuples or list of of floats
    from 0 to 1. These sequences must have a length of 1 (GRAY), 3 (RGB) or
    4 (CMYK). For GRAY colorspace, a single float instead of the unwieldy
    (float,) or [float] is also accepted. Accept (default) or use None to
    not use the parameter.
    """
    items: List[Any] = field(default_factory=list)
    "List of draw commands: lines, rectangles, quads or curves."
    lineCap: int = 0
    """Controls the look of line ends. The default value 0 lets each line
    end at exactly the given coordinate in a sharp edge. A value of 1 adds a
    semi-circle to the ends, whose center is the end point and whose diameter is
    the line width. Value 2 adds a semi-square with an edge length of line width
    and a center of the line end."""
    lineJoin: int = 0
    """
    Controls the way how line connections look like. This may be either as a
    sharp edge (0), a rounded join (1), or a cut-off edge (2, “butt”).
    """
    fill_opacity: float = 1
    """
    Float in range [0, 1]. Negative values or values > 1 will ignored (in most
    cases). Both set the transparency such that a value 0.5 corresponds to 50%
    transparency, 0 means invisible and 1 means intransparent. For e.g. a
    rectangle the stroke opacity applies to its border and fill opacity to its
    interior.
    """
    stroke_opacity: float = 1
    """
    Float in range [0, 1]. Negative values or values > 1 will ignored (in most
    cases). Both set the transparency such that a value 0.5 corresponds to 50%
    transparency, 0 means invisible and 1 means intransparent. For e.g. a
    rectangle the stroke opacity applies to its border and fill opacity to its
    interior.
    """
    rect: Rect = field(default_factory=Rect)
    layer: Optional[str] = None
    level: Optional[int] = None
    seqno: Optional[int] = None
    type: Optional[str] = None
    width: float = 0
    """
    The stroke (“border”) width of the elements in a shape (if applicable).
    The default value is 1. The values width, color and fill have the following
    relationship / dependency:
    * If fill=None shape elements will always be drawn with a border - even if
    color=None (in which case black is taken) or width=0 (in which case 1 is taken).
    * Shapes without border can only be achieved if a fill color is specified
    (which may be white of course). To achieve this, specify width=0. In this case,
    the color parameter is ignored.
    """

    def __post_init__(self):
        if not isinstance(self.lineCap, int):
            self.lineCap = int(max(self.lineCap))

    def finishing_opts(self) -> dict:
        """Dict, suitable for usage as argument for shape.finish()"""

        opts = {
            "width": self.width,
            "color": self.color,
            "fill": self.fill,
            "lineCap": self.lineCap,
            "lineJoin": self.lineJoin,
            "dashes": self.dashes,
            "closePath": self.closePath,
            "even_odd": self.even_odd,
            "stroke_opacity": self.stroke_opacity,
            "fill_opacity": self.fill_opacity,
        }
        return opts
    


@dataclass(frozen=True)
class ImageInfoPDF:
    number: int
    block_number: int
    rect: Rect
    width: int
    height: int
    cs_name: str
    colorspace: int
    xres: int
    yres: int
    bpc: int
    size: int
    digest: Optional[bytes] = None
    xref: int = 0
    transform: Matrix = field(default_factory=Matrix)
    has_mask: bool = False

    @classmethod
    def from_dict(cls, info: dict) -> "ImageInfoPDF":
        return cls(
            number=info.get("number", 0),
            block_number=info.get("block_number", 0),
            rect=Rect(*info.get("bbox", (0, 0, 0, 0))),
            width=info.get("width", 0),
            height=info.get("height", 0),
            cs_name=info.get("cs_name", ""),
            colorspace=info.get("colorspace", 0),
            xres=info.get("xres", 0),
            yres=info.get("yres", 0),
            bpc=info.get("bpc", 0),
            size=info.get("size", 0),
            digest=info.get("digest"),
            xref=info.get("xref", 0),
            transform=Matrix(*info.get("transform", (1, 0, 0, 0, 1, 0))),
            has_mask=info.get("has_mask", False),
        )


@dataclass
class TextSpanPDF:
    """
    Spans contain the actual text. A line contains more than one span only,
    if it contains text with different font properties.
    """
    rect: Rect
    "Span rectangle."
    origin: tuple[float, float]
    "The first character's origin (bottom-left corner of its bbox)."
    font: str
    "Font name."
    ascender: float
    """
    Ascender and descender can be used to compute the real text height.
    
    Ascender is the distance from the baseline to the top of the line for
    the font with size 1. To get real ascender in user space, multiply with font size.
    """
    descender: float
    """
    Ascender and descender can be used to compute the real text height.

    Descender is the distance from the baseline to the bottom of the line for
    the font with size 1. To get real descender in user space, multiply with font size.
    """
    size: float
    """
    Font size. It is smaller than the height of the span rectangle. To calculate
    the real rectangle for the text, use ascender and descender.
    """
    flags: int
    """
    Represents font properties except for the first bit 0. They are to
    be interpreted like this:
    * bit 0: superscripted (TEXT_FONT_SUPERSCRIPT) – not a font property, detected by MuPDF code.
    * bit 1: italic (TEXT_FONT_ITALIC)
    * bit 2: serifed (TEXT_FONT_SERIFED)
    * bit 3: monospaced (TEXT_FONT_MONOSPACED)
    * bit 4: bold (TEXT_FONT_BOLD)

    Test these characteristics like so:
    
    >>> if flags & pymupdf.TEXT_FONT_BOLD & pymupdf.TEXT_FONT_ITALIC:
            print(f"{span['text']=} is bold and italic")
    
    Bits 1 thru 4 are font properties, i.e. encoded in the font program. Please note, that 
    this information is not necessarily correct or complete: fonts quite often contain wrong
    data here.
    """
    char_flags: int
    """
    Represents extra character properties:
    * bit 0: strikeout.
    * bit 1: underline.
    * bit 2: synthetic (always 0, see char dictionary).
    * bit 3: filled.
    * bit 4: stroked.
    * bit 5: clipped.
    """
    color: tuple[float, float, float]
    "Text color in sRGB format."
    alpha: int
    "Alpha value (opacity) from 0 (transparent) to 255 (opaque)."
    text: str
    "The actual text of the span."

    def insert_text_dict(self) -> dict:
        """Return a dict suitable for insert_text() method of Shape object."""

        return {
            "point": self.origin,
            "buffer": self.text,
            "fontsize": self.size,
            "fontname": "helv",
            "color": self.color,
            "fill_opacity": self.alpha / 255.0,
            "stroke_opacity": self.alpha / 255.0,
            "encoding": 0
        }

    @classmethod
    def from_dict(cls, info: dict) -> "TextSpanPDF":
        color = info.get("color", (0, 0, 0))
        if isinstance(color, int):
            r = ((color >> 16) & 0xFF) / 255.0
            g = ((color >> 8) & 0xFF) / 255.0
            b = (color & 0xFF) / 255.0
            color = (r, g, b)
        return cls(
            rect=Rect(*info.get("bbox", (0, 0, 0, 0))),
            origin=tuple(info.get("origin", (0.0, 0.0))),
            font=info.get("font", ""),
            ascender=info.get("ascender", 0.0),
            descender=info.get("descender", 0.0),
            size=info.get("size", 0.0),
            flags=info.get("flags", 0),
            char_flags=info.get("char_flags", 0),
            color=color,
            alpha=info.get("alpha", 255),
            text=info.get("text", ""),
        )

@dataclass(frozen=True)
class TextLinePDF:
    rect: Rect
    "Line rectangle."
    wmode: int
    "Writing mode: 0 = horizontal, 1 = vertical."
    dir: tuple[float, float]
    "Direction unit vector."
    spans: List[TextSpanPDF]
    "List of spans in this line."

    @classmethod
    def from_dict(cls, info: dict) -> "TextLinePDF":
        spans = [TextSpanPDF.from_dict(span) for span in info.get("spans", [])]
        return cls(
            rect=Rect(*info.get("bbox", (0, 0, 0, 0))),
            wmode=info.get("wmode", 0),
            dir=tuple(info.get("dir", (1.0, 0.0))),
            spans=spans,
        )
    
    @property
    def text(self) -> str:
        return " ".join(span.text for span in self.spans)
    
    @property
    def rotation(self) -> int:
        base_dirs = {
            0: (1.0, 0.0),
            90: (0.0, -1.0),
            180: (-1.0, 0.0),
            270: (0.0, 1.0),
        }
        for angle in base_dirs:
            base_dirs[angle] = sum((a*b for a, b in zip(self.dir, base_dirs[angle]))) # type: ignore
        return max(base_dirs, key=base_dirs.get) # type: ignore


    def insert_text_dicts(self) -> List[dict]:
        """Return list of dicts suitable for insert_text() method of Shape object."""
        
        dcts = [span.insert_text_dict() for span in self.spans]
        for dct in dcts:
            dct.update({"rotate": self.rotation})
        return dcts

@dataclass(frozen=True)
class TextBlockPDF:
    rect: Rect
    "Block rectangle."
    lines: List[TextLinePDF]
    "List of lines in this block."

    @classmethod
    def from_dict(cls, info: dict, skip_blank_lines: bool = True) -> "TextBlockPDF":
        if info.get("type") != 0:
            raise ValueError(
                "TextBlock can only be created from a text block dictionary (type 0)."
            )
        lines: list[TextLinePDF] = []
        for line in info.get("lines", []):
            text_line = TextLinePDF.from_dict(line)
            if skip_blank_lines and text_line.text.isspace():
                continue
            lines.append(text_line)
        
        rect = Rect(*info.get("bbox", (0, 0, 0, 0)))
        if skip_blank_lines:
            rect = Rect()
            for line in lines:
                rect.include_rect(line.rect)
        return cls(
            rect=rect,
            lines=lines,
        )

    @property
    def text(self) -> str:
        return " ".join(line.text.strip() for line in self.lines)
    
    def insert_text_dicts(self) -> List[dict]:
        """Return list of dicts suitable for insert_text() method of Shape object."""
        
        dcts = []
        for line in self.lines:
            dcts.extend(line.insert_text_dicts())
        return dcts

    def __add__(self, other: "TextBlockPDF | TextLinePDF") -> "TextBlockPDF":
        """Summation of two TextBlock instances: concatenates lines and expands rect."""
        if isinstance(other, TextBlockPDF):
            # Combine lines
            new_lines = self.lines + other.lines
            # Expand rect to include both
            new_rect = Rect(self.rect)
            new_rect.include_rect(other.rect)
        elif isinstance(other, TextLinePDF):
            new_lines = self.lines + [other]
            new_rect = Rect(self.rect)
            new_rect.include_rect(other.rect)
        else:
            raise NotImplemented
        return TextBlockPDF(rect=new_rect, lines=new_lines)


class KeyedDict(dict):
    def __init__(self, factory):
        super().__init__()
        self.factory = factory

    def __missing__(self, key):
        value = self.factory(key)
        self[key] = value
        return value


@dataclass(frozen=True)
class Size:
    mean: float
    min: float
    max: float


@dataclass(frozen=True)
class FigureCaptionPDF:
    matched_pattern: str
    text: str
    font_props: int
    lines_no: int
    rect: Rect


@dataclass
class FigurePDF:
    rect: Rect
    caption: FigureCaptionPDF


