# artfinder

[![PyPI - Version](https://img.shields.io/pypi/v/artfinder.svg)](https://pypi.org/project/artfinder)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/artfinder.svg)](https://pypi.org/project/artfinder)

-----

## Table of Contents

- [artfinder](#artfinder)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
    - [Planned functionality](#planned-functionality)
    - [Implemented functionality](#implemented-functionality)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Search](#search)
  - [License](#license)

## Description

Artfinder is a tool for fetching scientific articles and extract data from them for analysis.

This tool is in very **early stage of development** and is probably not interesting or usefull for broad audience.
It is being developed as a hobby/recreational project.

My main goal is to make a systematic analysis in one particular (and rather narrow) field of science ([Laser-ablative synthesis of nanoparticles](https://en.wikipedia.org/wiki/Laser_ablation)). Therefore, I develop this tool mainly for assistance in achieving this particular goal.

### Planned functionality

- Search for scientific articles from open databases (Crossref, Pubmed, Google Scholar)
- Getting refernced and referencing articles
- Downloading articles *.pdf from open sources
- Extraction and recognition of scientific data, represented as plots and graphs in the scientific articles

### Implemented functionality

- Search for articles in Crossref database
- Downloading of articles *.pdf from open sources

## Installation

```console
pip install artfinder
```

For PDF captions printed inside figure images, optionally install the `tesseract`
command and its English language data (for example, `tesseract-ocr` and
`tesseract-ocr-eng` on Debian/Ubuntu). Artfinder uses OCR for captions at the
foot of those images; native captions do not require Tesseract. OCR may
misread small type and scientific symbols.

## Usage

### Search

```python
import artfinder

af = artfinder.Artfinder()
af.find_article(doi='10.1021/acs.chemrev.6b00468')
```

### Mark and export figures

```python
from artfinder.article_pdf import ArticlePDF

with ArticlePDF("supplement.pdf") as pdf:
    pdf.mark("figures", output_path="out")
    pngs = pdf.extract_figures(output_path="out")  # one PNG per figure label
```

When a sequence of panel letters such as (a–d) continues as (e–f) on the
next page, each page is marked separately. A repeated caption such as
`Figure 8. (continued)` also links crops on consecutive pages, even when the
panel letters are part of an image. The descriptive caption stays on its
original page. A page-scoped export selects by the primary crop’s page and
stacks all linked crops in reading order at one scale with a white gap.

## License

`artfinder` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

