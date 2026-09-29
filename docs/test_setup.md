# Running tests and type checks

`venv/` at the repo root is the working environment (Python 3.12). It holds `artfinder`
as an **editable install pointing at the main checkout's `src/`** — not at whatever
worktree you are currently in. Everything below exists to make sure you are checking the
code you just edited.

## Install the tooling first

Neither `pytest` nor `mypy` ships in `venv/`, nor do the pandas and requests stubs
that `hatch run types:check` fetches with `--install-types`:

```bash
venv/bin/pip install pytest mypy pandas-stubs types-requests
```

Without the stubs mypy reports every `import pandas` as an error.

## Tests

From the repo root **or from any worktree**:

```bash
venv/bin/python -m pytest tests
```

This is correct in a worktree with no extra setup: `[tool.pytest.ini_options]` in
`pyproject.toml` sets `pythonpath = ["src"]`, which pytest resolves relative to its
rootdir — the checkout you started it from — and places ahead of site-packages. Do not
remove that line; without it a worktree silently tests the main checkout's code, and
edits appear to have no effect.

Single file / single test:

```bash
venv/bin/python -m pytest tests/test_article_pdf/test_figure_captions.py -k test_figure_captions_keys_are_labels
```

If results look impossible — a new test failing on code you can see is correct, or an
edit changing nothing — confirm which source tree is loaded. This mirrors what the
`pythonpath` setting does for pytest:

```bash
PYTHONPATH=src venv/bin/python -c "import artfinder; print(artfinder.__file__)"
```

The path must be under the checkout you are working in. Note that the same command
*without* `PYTHONPATH=src` always reports the main checkout — that is the editable
install talking, and is expected rather than a symptom.

## Type checks

Always pass the **full targets**, as the hatch script does:

```bash
venv/bin/mypy src/artfinder tests
```

Whole-target runs resolve `artfinder.*` imports within the checkout, so this is correct
from a worktree with no environment variables. Checking a **single file** is not:

```bash
venv/bin/mypy src/artfinder/api.py          # WRONG in a worktree
PYTHONPATH=src venv/bin/mypy src/artfinder/api.py   # correct
```

Given one file, mypy resolves that file locally but follows its `artfinder.*` imports
through site-packages into the main checkout, mixing two copies of the package.
`MYPYPATH` does *not* fix this; `PYTHONPATH` does.

The hatch env is equivalent when `hatch` is installed:

```bash
hatch run types:check
```

## Known baseline

- `venv/bin/python -m pytest tests` — all tests pass.
- `venv/bin/mypy src/artfinder tests` — clean (2026-09-29), and clean with
  `--warn-unused-ignores` too.

PyMuPDF's `Page` is invisible to mypy: `pymupdf/__init__.py` binds the name to a string
before it defines the class. `article_pdf.py` therefore declares a stand-in `Page` under
`TYPE_CHECKING`, and reaches pages through `ArticlePDF._page(page_no)` rather than
`self.file[page_no]`, whose return type mypy cannot resolve. Use the accessor in new
code instead of adding `# type: ignore`.
