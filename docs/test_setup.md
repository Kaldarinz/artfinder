# Running tests and type checks

`venv/` at the repo root is the working environment (Python 3.12). It holds `artfinder`
as an **editable install pointing at the main checkout's `src/`** — not at whatever
worktree you are currently in. Everything below exists to make sure you are checking the
code you just edited.

## Install the tooling first

Install the test runner, parallel workers, numerical thread-pool control, type checker,
and the pandas and requests stubs that `hatch run types:check` fetches with
`--install-types`:

```bash
venv/bin/pip install pytest pytest-xdist threadpoolctl mypy pandas-stubs types-requests
```

The test dependencies are also declared in `pyproject.toml` as the `test` extra
(`pip install -e '.[test]'` for a fresh editable install). Do not repoint a shared
venv's editable install from a worktree; use the tooling-only command above there.

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

The default is **eight worker processes**, configured in `pyproject.toml` with
`pytest-xdist` and `--dist=worksteal`. Each golden PDF is a separate test case, so
slow documents can be distributed rather than leaving two corpus-wide loops as
serial bottlenecks. Missing golden PDFs fail instead of silently escaping coverage.
The expected captions and rectangles are unchanged.

Workers open their own PyMuPDF documents: no document is shared between processes.
The corpus is read-only, file-writing tests use isolated `tmp_path` directories or
unique temporary files, and HTTP tests use fake responses. `tests/conftest.py` limits loaded BLAS/OpenMP pools
to one native thread per process during tests, including serial runs, so eight
workers do not each start a machine-sized numerical pool. This does not change
library behavior outside pytest. Eight workers are not CPU affinity/pinning.

Serial debugging (also recommended for a single test):

```bash
venv/bin/python -m pytest -n 0 tests
venv/bin/python -m pytest -n 0 tests/test_article_pdf/test_figure_captions.py -k test_figure_captions_keys_are_labels
```

Override the worker count with `-n 4` on smaller machines. For the per-PDF golden
cases, use `-k` with part of the PDF filename to select documents.

### Local performance check (2026-11-07)

One full run per mode on the 20-logical-CPU development machine, Python 3.12.3:

| Mode | Result | Pytest elapsed time |
| --- | --- | --- |
| Original serial suite | 445 passed | 341.11 s |
| Per-PDF suite, eight workers (default) | 573 passed | 140.10 s |
| Per-PDF suite, `-n 0` | 573 passed | 417.83 s |

The test count increases because two loops become 130 cases (65 PDFs each), not
because any PDF or assertion was dropped. Both new modes ran the same 573 unique
cases with zero skips; corpus and golden-JSON hashes stayed unchanged. These are
local observations, not a controlled benchmark or a guaranteed speedup: other
workloads were active. The eight-worker run was about 2.4 times faster than the
original run, not eight times faster. One PDF,
`data-driven pre-determination of cu oxidation state in copper nanoparticles_si.pdf`,
took about 135–136 seconds per golden check in the parallel run and now sets the
critical path; splitting test cases cannot parallelize the inside of that check.

### Confirm the source tree

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
