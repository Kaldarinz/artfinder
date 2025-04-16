"""Module for helper functions."""

from __future__ import annotations

import datetime
import re
import sys
from ast import literal_eval

try:
    from IPython.display import DisplayHandle, display, clear_output
except ImportError:
    ...
from typing import (
    Generator,
    Optional,
    Union,
    cast,
    TypeVar,
    Callable,
    Coroutine,
    ParamSpec,
    Iterator,
    Any,
)
from xml.etree.ElementTree import Element as EtreeElement
import lxml.etree
from lxml.etree import _Element as LxmlElement
from typeguard import typechecked
from typing_extensions import TypeAlias
from pandas import DataFrame
import pandas as pd


Element: TypeAlias = Union[LxmlElement, EtreeElement]

T = TypeVar("T", bound=Optional[str])
P = ParamSpec("P")
TA = TypeVar("TA")


@typechecked
def arrange_query(
    search_term: str, start_date: datetime.date, end_date: datetime.date
) -> str:
    """
    Create a new query adding range date according PubMed format.

    Parameters
    ----------
    search_term : str
        term to search in PubMed database
    start_date : date
        Beginning of the range date.
    end_date : date
        Ending of the range date.

    Returns
    -------
    search query : str
        A new query to use in PubMed API.
    """
    date_filter = "Date - Publication"
    from_date = start_date.strftime("%Y/%m/%d")
    to_date = end_date.strftime("%Y/%m/%d")
    date_query = f'"{from_date}"[{date_filter}] : "{to_date}"[{date_filter}]'

    return f"({search_term}) AND ({date_query})"


@typechecked
def get_search_term(input_string: str) -> str:
    """
    Extract just the search term from a full query.

    by a full query is undertood a query with range date.

    Parameters
    ----------
    input_string : str
        A query with dates.

    Returns
    -------
    search term : str
        Isolated search term.

    Notes
    -----
    i.e.
    input_string:
    ('(virus AND bacteria) AND'
    ' ("2022/01/01"[Date - Publication] : "2023/12/01"[Date - Publication])')
    output:
    virus AND bacteria
    """
    # Regex pattern to match date ranges
    date_pattern = (
        r'AND\s*\(\s*"\d{4}/\d{2}/\d{2}"\s*\[Date - Publication\]\s*:'
        r'\s*"\d{4}/\d{2}/\d{2}"\s*\[Date - Publication\]\s*\)'
    )

    # Remove the date range
    cleaned_string = re.sub(date_pattern, "", input_string)

    # Ensure no extra spaces around the logical operators
    cleaned_string = re.sub(r"\s+", " ", cleaned_string).strip()

    # Remove leading and trailing parentheses if
    # they enclose the whole expression
    if cleaned_string.startswith("(") and cleaned_string.endswith(")"):
        cleaned_string = cleaned_string[1:-1]

    return cleaned_string.strip()


@typechecked
def get_range_months(
    start_date: datetime.date, end_date: datetime.date
) -> list[tuple[datetime.date, datetime.date]]:
    """
    Divide a range date into month ranges.

    Parameters
    ----------
    start_date: date
        Beginning of the range date.
    end_date: date
        Ending of the range date.

    Returns
    -------
    date_ranges: list[tuple[date, date]]
        A list of smaller range dates per month.

    Notes
    -----
    i.e. (in date python type)
    input: 2020-03-01 to 2020-05-15
    output:
        [(2020-03-01, 2020-03-31),
         (2020-04-01, 2020-04-30),
         (2020-05-01, 2020-05-15)]
    """
    MONTHS_IN_YEAR = 12
    date_ranges = []

    # Start at the first day of the starting month
    current_start_date = start_date

    while current_start_date <= end_date:
        # Calculate the end of the current month
        # i.e. current_start_date = 2020-03-01
        # Showing a simulation of how are changing the variables
        # next_month = 2020-03-29 + 000-00-04 = 2020-04-02
        # current_end_date = 2020-04-02 - 000-00-02 = 2020-03-31
        next_month = current_start_date.replace(day=28) + datetime.timedelta(days=4)
        current_end_date = next_month - datetime.timedelta(days=next_month.day)

        # Adjust the end date if it goes beyond the specified end_date
        current_end_date = min(current_end_date, end_date)

        date_ranges.append((current_start_date, current_end_date))

        # Move to the first day of next month
        if current_end_date.month == MONTHS_IN_YEAR:
            current_start_date = current_end_date.replace(
                year=current_end_date.year + 1, month=1, day=1
            )
        else:
            current_start_date = current_end_date.replace(
                month=current_end_date.month + 1, day=1
            )

    return date_ranges


@typechecked
def get_range_date_from_query(
    input_string: str,
) -> tuple[datetime.date, datetime.date] | None:
    """
    Extract the dates from a PubMed query.

    If there no range date return None.

    Parameters
    ----------
    input_string: str
        A PubMed query

    Returns
    -------
    date_ranges: tuple[date, date]
        A tuple with the start range date and end range date.

    Notes
    -----
    i.e.
    input_string:
    ('(virus AND bacteria) AND'
    ' ("2022/01/01"[Date - Publication] : "2023/12/01"[Date - Publication])')
    output:
    (date(2022/01/01), date(2023/12/01)
    """
    EXPECTED_DATES = 2
    # Regular expression to match the date pattern
    date_pattern = r'"(\d{4}/\d{2}/\d{2})"\[Date - Publication\]'

    # Find all matches of the date pattern in the input string
    matches = re.findall(date_pattern, input_string)

    # Convert the extracted date strings into datetime objects
    dates = [datetime.datetime.strptime(date, "%Y/%m/%d").date() for date in matches]

    # Check if exists dates otherwise return None
    if len(dates) == EXPECTED_DATES:
        return (dates[0], dates[1])

    return None


@typechecked
def get_range_years(
    start_date: datetime.date, end_date: datetime.date
) -> list[tuple[datetime.date, datetime.date]]:
    """
    Divide a range date into year range.

    Parameters
    ----------
    start_date: date
        begining of the range date.
    end_date: date
        ending of the range date.

    Returns
    -------
    date_ranges: list[tuples[date, date]]
        a list of a smaller range dates per year.

    Notes
    -----
    i.e. date type

    input: 2020-03-01 to 2022-12-23

    output:
    [(2020-03-01, 2020-12-31),
    (2021-01-01, 2021-12-31),
    (2022-01-01, 2022-12-23)]
    """
    # Define the end of the starting year
    date_ranges = []

    end_of_first_year = datetime.datetime(start_date.year, 12, 31).date()

    if start_date <= end_of_first_year:
        # If the start date is before the end of the year, add this period
        # to the list
        date_ranges.append((start_date, min(end_of_first_year, end_date)))

    # Iterate over the full years between the start and end dates
    for year in range(start_date.year + 1, end_date.year):
        start_of_year = datetime.datetime(year, 1, 1).date()
        end_of_year = datetime.datetime(year, 12, 31).date()
        date_ranges.append((start_of_year, end_of_year))

    # Define the start of the ending year
    if end_date.year >= start_date.year:
        start_of_last_year = datetime.date(end_date.year, 1, 1)
        if start_of_last_year < end_date:
            date_ranges.append((start_of_last_year, end_date))

    return date_ranges


@typechecked
def batches(iterable: list[str], n: int = 1) -> Generator[list[str], None, None]:
    """
    Create batches from an iterable.

    Parameters
    ----------
    iterable: Iterable
        the iterable to batch.
    n: Int
        the batch size.

    Returns
    -------
    batches: List
        yields batches of n objects taken from the iterable.
    """
    # Get the length of the iterable
    length = len(iterable)

    # Start a loop over the iterable
    for index in range(0, length, n):
        # Create a new iterable by slicing the original
        yield iterable[index : min(index + n, length)]


@typechecked
def getContent(
    element: Element,
    path: str,
    default: T = None,
    separator: str = "\n",
) -> T:
    """
    Retrieve text content of an XML element.

    Parameters
    ----------
    element: Element
        the XML element to parse.
    path: Str
        Nested path in the XML element.
    default: Str
        default value to return when no text is found.

    Returns
    -------
    text: Str
        text in the XML node.
    """
    # Find the path in the element
    result = element.findall(path)

    # Return the default if there is no such element
    if result is None or len(result) == 0:
        return default

    # Extract the text and return it
    return cast(T, separator.join([sub.text for sub in result if sub.text is not None]))


@typechecked
def getContentUnique(
    element: Element,
    path: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Retrieve text content of an XML element. Returns a unique value.

    Parameters
    ----------
    element: Element
        the XML element to parse.
    path: Str
        Nested path in the XML element.
    default: Str
        default value to return when no text is found.

    Returns
    -------
    text: Str
        text in the XML node.
    """
    # Find the path in the element
    result = cast(list[Element], element.findall(path))

    # Return the default if there is no such element
    if not result:
        return default

    # Extract the text and return it
    return cast(str, result[0].text)


@typechecked
def getAllContent(
    element: Element,
    path: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Retrieve text content of an XML element.

    Return all the text inside the path and omit XML tags inside.

    Parameters
    ----------
    element: Element
        the XML element to parse.
    path: Str
        Nested path in the XML element.
    default: Str
        default value to return when no text is found.

    Returns
    -------
    text: str
        text in the XML node.
    """
    # Find the path in the element
    raw_result = element.findall(path)

    # Return the default if there is no such element
    if not raw_result:
        return default

    # Get all text avoiding the tags
    raw_bytes = lxml.etree.tostring(
        raw_result[0],  # type: ignore
        method="text",
        encoding="utf-8",
    )

    if isinstance(raw_bytes, bytes):
        result = raw_bytes.decode("utf-8")
    else:
        result = raw_bytes

    # Extract the text and return it
    return " ".join(result.split())


@typechecked
def getAbstract(
    element: Element,
    path: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Retrieve text content of an XML element.

    Return all the text inside the path and omit XML tags inside.
    and omits abstract-type == scanned-figures

    Parameters
    ----------
    element: Element
        the XML element to parse.
    path: Str
        Nested path in the XML element.
    default: Str
        default value to return when no text is found.

    Returns
    -------
    text: str
        text in the XML node.
    """
    # Find the path in the element
    raw_result = element.findall(path)

    # Return the default if there is no such element
    if not raw_result:
        return default

    if raw_result[0].attrib.get("abstract-type", None) == "scanned-figures":
        return default

    for fig in raw_result[0].iter("fig"):
        parent = fig.getparent()  # type: ignore
        if parent is not None:
            parent.remove(fig)  # type: ignore

    # Get all text avoiding the tags
    raw_bytes = lxml.etree.tostring(
        raw_result[0],  # type: ignore
        method="text",
        encoding="utf-8",
    )
    if isinstance(raw_bytes, bytes):
        result = raw_bytes.decode("utf-8")
    else:
        result = raw_bytes

    # Extract the text and return it
    return " ".join(result.split())


@typechecked
def _print_tags_recursively(element: LxmlElement, indent=0):
    """
    Print a human-readable representation of the tags in the XML element recursively.

    Parameters
    ----------
    element: xml.Element
        The XML element to print the tags from.
    indent: int
        The indentation level for pretty printing.
    """
    print("|-" * indent + f"Tag: {element.tag}: {element.text}")
    for child in element:
        _print_tags_recursively(child, indent + 1)


@typechecked
def pretty_print_xml(xml: LxmlElement) -> None:
    """
    Print a human-readable representation of the root tags in the XML response.

    Parameters
    ----------
    xml_response: str
        The XML response as a string.
    """

    print("Root tags:")
    _print_tags_recursively(xml)


class LinePrinter:
    """
    A utility class for printing text on the same line in the terminal.

    This class is useful for dynamically updating a single line of output,
    such as progress indicators or status messages.

    Usage:
    ------
    ```python
    printer = LinePrinter()
    printer("Processing...")
    printer("Completed!")
    printer.close()
    ```

    This will overwrite "Processing..." with "Completed!" on the same line.

    Notes:
    ------
    - In Jupyter Notebook environments, it uses IPython's `DisplayHandle` to update the output.
    - In terminal environments, it uses ANSI escape codes to overwrite the current line.
    """

    def __init__(self) -> None:
        if "ipykernel" in sys.modules:
            self.display_id = cast(DisplayHandle, display(display_id=True))

    def __call__(self, text) -> None:
        if "ipykernel" in sys.modules:
            self.display_id.update(text)
        else:
            print("\033[2K\033[1G", end="")
            print(text, end="", flush=True)

    def close(self) -> None:
        if "ipykernel" not in sys.modules:
            print()

    def clear(self) -> None:
        self('')
        self.close()

    def __enter__(self) -> LinePrinter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        if exc_type is not None:
            raise exc_value


class MultiLinePrinter:
    """
    A class for handling dynamic multi-line printing that works in both terminal and Jupyter environments.

    MultiLinePrinter manages multiple lines of text that can be updated in-place, with different
    implementations for terminal environments (using ANSI escape codes) and Jupyter notebooks
    (using IPython's display functionality).

    Designed for printing progress during async callls.

    Parameters
    ----------
    lines : int
        The number of lines to manage.

    Attributes
    ----------
    lines_no : int
        The total number of lines managed by the printer.
    lines : List[PrinterLine]
        The list of PrinterLine objects representing each line.
    first_run : bool
        Flag indicating if this is the first print operation.
    display_id : DisplayHandle, optional
        Handle for display in Jupyter environments.

    Usage:
    ------
    ```python
    printer = MultiLinePrinter(2)
    line1 = printer.get_line()
    line1("Processing line 1...")
    with printer.get_line() as line2:
        line2("Processing line 2...")
    printer.close()
    """

    def __init__(self, lines: int) -> None:
        self.lines_no = lines
        self.lines = [PrinterLine(i, False, self) for i in range(lines)]
        self.first_run = True

    def print(self) -> None:
        if "ipykernel" in sys.modules:
            if not hasattr(self, "display_id"):
                self.display_id = cast(DisplayHandle, display(display_id=True))
            clear_output(wait=True)
            print(
                "\n".join(
                    (self.lines[i].text for i in range(self._max_non_empty_index() + 1))
                )
            )

        else:
            if not self.first_run:
                # clear lines
                print(
                    f"\033[{self._max_non_empty_index() + 1}A\033[1G\033[0J",
                    end="",
                    flush=True,
                )
            # print lines
            for i in range(self._max_non_empty_index() + 1):
                print(self.lines[i].text)
            print(self.lines[-1].text, end="", flush=True)
            self.first_run = False

    def _max_non_empty_index(self) -> int:
        """
        Get the index of the last busy line.
        """
        for i in range(self.lines_no - 1, -1, -1):
            if self.lines[i].text:
                return i
        return -1

    def get_line(self) -> PrinterLine:
        for line in self.lines:
            if not line.busy:
                line.busy = True
                return line
        raise RuntimeError("No available lines")

    def free_line(self, line: PrinterLine) -> None:
        line.busy = False

    def close(self) -> None:
        if "ipykernel" not in sys.modules:
            print()
        else:
            print()


class PrinterLine:

    def __init__(self, id: int, busy: bool, printer: MultiLinePrinter) -> None:
        """
        Initialize a PrinterLine instance.

        Parameters
        ----------
        id : int
            The unique identifier for the printer line.
        busy : bool
            Indicates whether the line is currently in use.
        printer : MultiLinePrinter
            The parent MultiLinePrinter instance managing this line.
        """
        self.id = id
        self.busy = busy
        self.text = ""
        self.printer = printer

    def __call__(self, text: str) -> None:
        self.text = text
        self.printer.print()

    def free(self) -> None:
        self.busy = False

    def update(self, text: str) -> None:
        """
        Only update line text, do not print it.
        """
        self.text = text

    def __enter__(self) -> PrinterLine:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.free()
        if exc_type is not None:
            raise exc_value


def strict_filter(title: str) -> bool:

    # patterns
    parts = [
        r"(?=.*laser\w*)(?=.*\w*(gener|synth|prod|manufact|fabric)\w*)(?=.*(nano|colloid|quantum\sdot)\w*|.*\bnps\b)",
        r"(?=.*(nano|particle|cluster)\w*)(?=.*\b\w*(ablat|fragment)\w*)",
    ]
    pattern = r"(" + r"|".join(parts) + r")"
    # exclude patterns
    exlude_parts = [
        r"(?!.*nanostructur(ing|ed)\w*)",
    ]
    pattern += r"".join(exlude_parts)
    return re.search(pattern, title, re.IGNORECASE) is not None
