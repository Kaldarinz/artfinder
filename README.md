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
* Search for scientific articles from open databases (Crossref, Pubmed, Google Scholar)
* Getting refernced and referencing articles
* Downloading articles *.pdf from open sources
* Extraction and recognition of scientific data, represented as plots and graphs in the scientific articles

### Implemented functionality
* Search for articles in Crossref database
* Downloading of articles *.pdf from open sources

## Installation

```console
pip install artfinder
```

## Usage

### Search
```python
import artfinder

af = artfinder.Artfinder()
af.find_article(doi='10.1021/acs.chemrev.6b00468')
```

## License

`artfinder` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.