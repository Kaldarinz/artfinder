"""Independent pixel assertions for figure exports."""

from collections.abc import Sequence
from pathlib import Path

from pymupdf import Pixmap


def assert_stacked_png(path: Path, panels: Sequence[Pixmap], gap: int) -> None:
    """Compare the entire PNG with the centered source crops, including white padding.

    Parameters
    ----------
    path : Path
        Exported PNG.
    panels : Sequence[Pixmap]
        Independently rendered source crops in reading order.
    gap : int
        Expected number of white rows between crops.
    """
    out = Pixmap(str(path))
    assert out.width == max(panel.width for panel in panels)
    assert out.height == sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    pixels = bytearray(b"\xff") * (out.stride * out.height)
    top = 0
    for panel in panels:
        assert out.n == panel.n
        left = (out.width - panel.width) // 2
        data = panel.samples_mv
        for row in range(panel.height):
            start = (top + row) * out.stride + out.n * left
            pixels[start:start + panel.stride] = data[row * panel.stride:(row + 1) * panel.stride]
        top += panel.height + gap
    assert out.samples_mv == pixels
