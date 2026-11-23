# Remaining work on `ArticlePDF`

Written 2026-09-10, after the figure-label / supplementary-information work. Each
item states what is wrong, what it costs, and what was already tried, so none of
it has to be rediscovered.

## Known defects

### `line_pitch` is wrong where a block holds more than one line per row

`line_pitch` is the median gap between consecutive lines *inside* a text block.
PyMuPDF emits sub- and superscripts (`R²`, `P₁₂`) and cells printed side by side
as separate lines of the same block, so those gaps are zero or a fragment offset
rather than the leading.

In `data-driven pre-determination of cu oxidation state in copper nanoparticles_si.pdf`
the pitch measures **6.09 pt where the real spacing is ~20.6 pt** (866 within-block
gaps, median 0.00). Both the caption stitching and the paragraph stitching use the
pitch, so on that document a paragraph's last line is not absorbed and the figure
below it swallows the line — its recorded rectangles are today's behaviour, not
the right answer, and must be re-recorded once this is fixed.

Measured and rejected:

- gaps guarded by line height (`gap > 0.5 * height`) — 6.13, no better;
- consecutive baselines across the whole page — 3.30, worse, because a two-column
  page interleaves the baselines of both columns; also moves one golden.

A fix has to be column-aware: measure within one column, or over blocks of
paragraph width only.

### A supplementary file with no body text has no paragraph width

`boron nanoparticle-enhanced proton therapy for cancer treatment_si.pdf` holds
eight text blocks over three pages, six of them prose, all of different widths, so
DBSCAN calls every one noise and `paragraph_width` is `(0, 0, 0)`.
`columns_number` now assumes one column rather than raising, but every heuristic
that leans on the width is running blind: nothing bounds a figure from above
except the header band, and `S1` accordingly includes the "Supplementary
materials" heading.

A fallback width — the widest prose block, or the page width less its margins —
would probably serve better than zero. It changes behaviour wherever the width is
currently zero, so it needs measuring first.

### The caption font filter loses the smaller of two caption sets

`_find_captions` keeps only the captions whose first-span font is the most common
in the document. In `whole-cell patch-clamp measurements of spermatozoa reveal an
alkaline-activated ca channel_with_si.pdf` the four main captions are
`AdvPS4DD239` and the three supplementary ones `Arial-BoldMT`, so only `1`–`4`
are found. The fixture records this deliberately.

Fixing it means grouping captions by font and keeping every group that looks like
a caption run, rather than the single most common font.

### The grid-coarsening fallback can manufacture a band

`_repeating_rects_in` lowers the clipping precision until *something* repeats on
`header_min_pages` pages. Where a document has no running header or footer, that
degrades until unrelated rectangles collide and a band is declared anyway. It was
behind the old ACS Nano false header, and it is why `exciton photoluminescence…`
briefly gained a footer at y=710 during the intersection work.

Turning the fallback off costs three header/footer detections and moves one
golden, so it needs its own measurement rather than being bundled into another
change.

### `_refine_figure_rect` requires containment

An element that straddles the figure rectangle is dropped rather than clipped,
which turns a small vertical error into a collapsed rectangle. This is what made
the ACS Nano figures collapse onto a line number before the header rule was
fixed. The header fix removed the trigger; the sharp edge remains. Clipping
partially overlapping elements instead moves 3 of 39 goldens.

### Paragraph stitching could absorb a caption

`_get_paragraphs_from_page` grows a paragraph over any block one `line_pitch`
below it that overlaps it horizontally. A figure caption printed exactly one pitch
under a paragraph would be absorbed. No fixture does it — captions sit further
away — but it is the failure mode this rule introduced.

## Features not started

- **Table cell extraction.** `tables` gives a rectangle and a caption; the cell
  grid is not extracted. The row/column structure is already visible in the block
  lines (cells of a row share a y range at repeating x positions), so a
  `to_pandas()`-style reader has something to build on.
- **`is_manuscript`.** `_line_number_rects` is computed and then used only to
  delete lines. Line numbering means a manuscript or preprint rather than a
  typeset article, which predicts single-column layout, generous leading, and main
  text and supplementary information bound together. Cheap to expose, and useful
  to `ablation_db`.

## Housekeeping

- `ablation_db` still calls the `int`-keyed figure API and needs updating for the
  string labels (`figures`, `figure_captions`, `figure_label`, `extract_figures`
  returning a mapping).
- `tselikov-et-al-2022-…pdf` is in the fixture directory but in neither golden
  file. Its header used to be a logo artifact; that is fixed, so it could be
  recorded now.
- `_calc_paragraph_width` still warns `No clusters found (all points are noise)`
  through `warnings.warn`. For a supplementary file that is a normal outcome, and
  the header and footer report the same situation at debug level.
