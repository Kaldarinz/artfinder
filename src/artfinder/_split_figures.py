"""Panel-letter evidence for a figure split at a page break."""

from collections.abc import Sequence

from pymupdf import Rect

PanelMark = tuple[str, Rect]


def _rows(marks: Sequence[PanelMark]) -> list[list[PanelMark]]:
    """Read standalone panel letters left to right on each visual row."""
    rows: list[list[PanelMark]] = []
    for mark in sorted(marks, key=lambda item: item[1].y0):
        rect = mark[1]
        if rows:
            first = rows[-1][0][1]
            if first.y1 - rect.y0 >= 0.5 * min(first.height, rect.height):
                rows[-1].append(mark)
                continue
        rows.append([mark])
    for row in rows:
        row.sort(key=lambda item: item[1].x0)
    return rows


def continues_panel_sequence(before: Sequence[PanelMark], after: Sequence[PanelMark]) -> bool:
    """Require successive letters from (a) and aligned panels across the break.

    Parameters
    ----------
    before, after : Sequence[PanelMark]
        Panel letters and their rectangles on the preceding and caption pages.

    Returns
    -------
    bool
        Whether the letters form a single sequence in page and row order.
    """
    if not before or not after:
        return False
    previous, following = _rows(before), _rows(after)
    letters = [letter for row in previous + following for letter, _ in row]
    if any(ord(letter) != ord("a") + i for i, letter in enumerate(letters)):
        return False
    # Compare the shared columns of the rows meeting at the break. A short
    # final row need not contain as many panels as the row before it.
    return all(abs(left.x0 - right.x0) <= max(left.height, right.height)
               for (_, left), (_, right) in zip(previous[-1], following[0]))
