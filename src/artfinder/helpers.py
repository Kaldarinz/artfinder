# SPDX-FileCopyrightText: 2025-present Anton Popov <a.popov.fizteh@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Module for helper functions."""

from __future__ import annotations

import re
import sys

try:
    from IPython.display import DisplayHandle, display
except ImportError:
    ...
from typing import (
    Optional,
    cast,
    TypeVar,
    ParamSpec,
)

T = TypeVar("T", bound=Optional[str])
P = ParamSpec("P")

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
        if "ipykernel" in sys.modules:
            print()

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
    max_line_len : int
        The maximum length of each line. 
        Used to truncate long lines in terminal environments.

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

    def __init__(self, lines: int, max_line_len: int = 80) -> None:
        if "ipykernel" in sys.modules:
            display(clear=True)
            self.display_id = cast(DisplayHandle, display(display_id=True))
        self.lines_no = lines
        self.lines = [PrinterLine(i, False, self) for i in range(lines)]
        self.first_run = True
        self.max_line_len = max_line_len
        self.num_printed_lines = 0

    def print(self) -> None:
        if "ipykernel" in sys.modules:
            # IDK, this is the only way to make it work
            self.display_id.update([self._format_line(self.lines[i]) for i in range(self._max_non_empty_index() + 1)], clear=True)

        else:
            if not self.first_run:
                # clear printed lines if not first run
                print(
                    f"\033[{self.num_printed_lines}A\033[1G\033[0J",
                    end="",
                    flush=True,
                )
            # print lines
            for i in range(self._max_non_empty_index() + 1):
                line_text = self._format_line(self.lines[i])
                print(line_text)
            line_text = self._format_line(self.lines[-1])
            print(line_text, end="", flush=True)
            self.num_printed_lines = self._max_non_empty_index() + 1
            self.first_run = False


    def _format_line(self, line: 'PrinterLine') -> str:
        """
        Format the line text to fit within the maximum line length.
        """
        if len(line.text) > self.max_line_len:
            return line.text[: self.max_line_len] + "..."
        return line.text

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
        if "ipykernel" in sys.modules:
            pass
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
