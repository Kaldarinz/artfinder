# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## What this is

`artfinder` fetches scientific article metadata (Crossref), downloads open-access PDFs, and extracts figures/captions from those PDFs. Hobby project targeting one research field (laser-ablative nanoparticle synthesis); early stage, single maintainer, no CI.

Packaging: hatchling, `src/` layout, `requires-python >= 3.10`. Version lives in `src/artfinder/__about__.py`.

## Environment & commands

`venv/` at the repo root is the working environment (Python 3.12) with `artfinder` installed editable **from the main checkout**. **Neither `pytest` nor `hatch` is installed there** — install before running tests or type checks:

```bash
venv/bin/pip install pytest
```

Run the tests:

```bash
venv/bin/python -m pytest tests
```

Single test file / single test:

```bash
venv/bin/python -m pytest tests/test_article_pdf/test_figure_captions.py -k test_figure_captions_keys_are_integers
```

Type check (defined as a hatch env in `pyproject.toml`; needs `hatch`, or run `mypy` directly against the same targets after `venv/bin/pip install mypy`):

```bash
hatch run types:check
# or
venv/bin/mypy src/artfinder tests
```

Before completing a change, type-check every edited Python file and resolve reported errors. The type check is *not* clean — see the baseline in `docs/test_setup.md` and make sure your change does not raise the total.

**Working in a `git worktree`?** The venv's editable install points at the main checkout, so a careless invocation checks the wrong source tree and your edits look like no-ops. Tests are already handled (`pythonpath = ["src"]` in `pyproject.toml`), but single-file `mypy` runs are not. Read [docs/test_setup.md](docs/test_setup.md) before running anything from a worktree.

`tst.py` is a scratch script (gitignored via `tst*`) — safe to overwrite for quick manual checks.

## Architecture

Two largely independent halves that meet only in `api.ArtFinder`:

**1. Bibliographic metadata (Crossref)** — `api.py` → `crossref.py` → `http_requests.py` → `article.py`

- `ArtFinder` (`api.py`) is the user-facing facade: `search`, `isearch` (count only), `find_article`, `get_refs`, `download_pdf`, `get_journal_info`, `get_white_list_data`. Everything returns pandas `DataFrame`/`Series`, not objects. Only Crossref is implemented; `database="pubmed"/"all"` raises `NotImplementedError`.
- `Crossref` (subclass of abstract `Endpoint`) is a fluent query builder: `.search().author().filter().article().get_df()`. Note the chaining is *not* immutable — each method mutates `self.request_params` in place and then returns `self.from_self()`, a new instance built from the same shared params. Filter kwarg names use underscores and are converted to Crossref's dashes (`from_pub_date` → `from-pub-date`); values are checked by `CrossrefFilterValidator.VALIDATORS`, so adding a new filter means adding it there too.
- `Endpoint.__iter__` handles pagination: if `rows`/`sample` is set it does a single request, otherwise it cursor-paginates in `ROW_LIMIT`-sized batches.
- `AsyncHTTPRequest` (`http_requests.py`) owns rate limiting: a semaphore for concurrency, a spacing delay derived from `CrossrefRateLimit`, and 429 handling via a shared `asyncio.Event` that pauses all in-flight tasks. Rate limits are re-read from `x-rate-limit-*` response headers on every 200.
- `FileDownloader` downloads PDFs concurrently, classifying results into `downloaded`/`restricted` (403)/`missing` (404)/`failed`, and detects HTML CAPTCHA pages served with a 200.

**2. PDF figure extraction** — `article_pdf.py` (the largest module) + the `*PDF` dataclasses in `dataclasses.py`

`ArticlePDF` wraps a PyMuPDF `Document`. Page-level extraction results are memoized in `KeyedDict` caches (`_text_cache`, `_drawings_cache`, `_images_cache`, `_figures_cache`) keyed by page number; document-level derived values are `@cached_property`. Raw PyMuPDF dicts are converted into typed dataclasses (`TextBlockPDF`/`TextLinePDF`/`TextSpanPDF`, `DrawingObjectPDF`, `ImageInfoPDF`) via `from_dict`.

The figure-detection pipeline is heuristic and layered — changing an early stage silently shifts everything downstream:

1. `_calc_paragraph_width` clusters text-block widths with DBSCAN; the heaviest cluster defines "body paragraph" width. `columns_number` and `get_paragraph_rects` are both derived from it.
2. `_find_figure_captions` regex-matches `Fig./Figure N`, stitches captions split across blocks/lines, then keeps only captions whose first-span font properties match the document's most common — this is what separates real captions from body paragraphs starting with "Fig.".
3. `_get_figures_from_page` derives each figure's bounding box from its caption: bottom = caption top, top = the nearest paragraph/other caption/header above, sides narrowed by column layout. Side captions (narrow + multiline) take a separate path that assumes full page width.
4. `extract_figures` rasterizes the resulting rects to PNG; `mark` writes an annotated copy of the PDF with detected elements outlined (useful for debugging the heuristics visually).

**Journal ranking** — `scimagojr.py` reads `data/scimagojr_*.csv` (semicolon-separated, comma decimal) and matches by title, falling back to substring match on the ISSN column. `white_list.py` queries `journalrank.rcsi.science` for the Russian journal white-list level. `ArtFinder.get_journal_info` special-cases SPIE proceedings by hardcoding the Proc. SPIE ISSNs.

## Things that will bite you

- **Everything sync wraps async in a thread.** `http_requests._execute_coro` and `white_list._run_coro_sync` both run coroutines in a fresh `Thread` + `asyncio.run` rather than calling `asyncio.run` directly. This is deliberate: the library is used from Jupyter, where an event loop is already running. Don't "simplify" it.
- **DataFrames round-trip through strings.** `Article.to_dict` stringifies and lowercases every field; `article._format_df` reconstructs list/dict columns (`authors`, `references`, `issn`, `links`, `funders`, `keywords`, `license`) with `ast.literal_eval` after patching `"none"` back to `"None"`. So list-valued columns hold real Python lists in memory but are always parsed back from text — hence `load_csv` works, and hence `issn` is a `list[str]`, never a scalar or NaN.
- **`SciMagoJR.BASE_PATH` is `Path(__file__).parent.parent.parent / "data"`**, i.e. the repo root's `data/`. This resolves correctly only from a source checkout / editable install; the wheel ships `data/**` but the path does not point at it.
- **Status output is a custom terminal printer.** `LinePrinter`/`MultiLinePrinter` (`helpers.py`) use ANSI escapes in a terminal and IPython `DisplayHandle` under `ipykernel`. `print_status=False` on `ArtFinder` disables it. Plain `print()` in these code paths will corrupt the display.
- **PDF tests are golden-file tests.** `tests/test_article_pdf/article_pdfs/` holds ~35 real PDFs with expected output in `all_fig_captions.json` and `figure_rects.json`. Tuning any heuristic in `article_pdf.py` means re-checking those goldens — do not regenerate them wholesale to make tests pass.
- `Article` and `CrossrefArticle` use `__slots__`; `get_all_slots()` walks the MRO and defines the canonical DataFrame column set.

## Python style

- Fully annotate every function and method, including private helpers and test fixtures.
- Prefer PEP 604 unions (`X | None`), `Self` for same-class returns, and precise container types over `Any`. Narrow `Any` with `cast` as early as practical.
- Use f-strings for all interpolation, including logging calls.
- Use NumPy-style docstrings (already the convention throughout this codebase — see `api.py`, `crossref.py`): one-line summary, then `Parameters`, `Raises` (only for exceptions explicitly raised), and `Returns` (only for non-`None` returns) sections, each preceded by exactly one blank line.

## Plans

- Write any non-trivial implementation plan as a Markdown file under `docs/plans/`, not in the chat alone. State at the top of the plan that it is deleted once every step is implemented — this repo has no `docs/` folder yet, so creating one is fine when a plan first needs it.
- Delete the plan file as part of the change that completes it.

## Git workflow

- If asked to work on more than one task at once, or the user asks for a worktree, use `git worktree` so agents don't interfere with each other's changes. Do not copy changes from a worktree back into another checkout.
- Leave changes uncommitted in the working tree unless explicitly asked to commit. The user reviews and commits manually.
- Preserve unrelated user changes. Never revert, stash, or overwrite modifications you did not make.
- Never create commits or pull requests unprompted, and do not offer to. Suggest a commit message only when asked: one concise summary line, plus up to four or five lines of additional description when useful.
