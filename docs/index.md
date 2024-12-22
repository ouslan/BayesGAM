# BayesGAM Documentation

![Image title](assets/bayesgam_tensor.png){ align=left }

![Build Status](https://travis-ci.org/dswah/BayesGAM.svg?branch=master) ![Coverage](https://codecov.io/gh/dswah/BayesGAM/branch/master/graph/badge.svg)  ![PyPi Version](https://badge.fury.io/py/bayesgam.svg)  ![Py27](https://img.shields.io/badge/python-2.7-blue.svg)  ![Py36](https://img.shields.io/badge/python-3.6-blue.svg)  ![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.1208723.svg)  ![Open Source](https://img.shields.io/badge/powered%20by-Open%20Source-orange.svg?style=flat&colorA=E1523D&colorB=007D8A) 

BayesGAM is a fork of pyGAM with the addition of Bayessian estimations. This is a dropin replacement of pyGAM. This is a package for building Generalized Additive Models in Python, with an emphasis on modularity and performance. The API will be immediately familiar to anyone with experience of scikit-learn or scipy.

## Installation

BayesGAM is on PyPi, and can be installed using `pip`:

```bash
pip install bayesgam
```

Alternatively, you can install it via `conda-forge` (although this is typically less up-to-date):

```bash
conda install -c conda-forge BayesGAM
```

To install the bleeding edge version from GitHub using `uv`:

1. Clone the repo and navigate to the main directory.
2. Run the following commands:

```bash
pip install uv
uv build
```

### Optional

To speed up optimization on large models with constraints, it helps to have `scikit-sparse` installed because it contains a slightly faster, sparse version of Cholesky factorization. The import from `scikit-sparse` references `nose`, so you'll need that too.

The easiest way is to use Conda:

```bash
conda install -c conda-forge scikit-sparse nose
```

More information is available in the [scikit-sparse docs](http://pythonhosted.org/scikit-sparse/overview.html#download).

## Dependencies

BayesGAM is tested on Python 2.7 and 3.6 and depends on `NumPy`, `SciPy`, and `progressbar2` (see `requirements.txt` for version information).

Optional: `scikit-sparse`.

In addition to the above dependencies, the `datasets` submodule relies on `Pandas`.

## Citing BayesGAM

Servén D., Brummitt C. (2018). pyGAM: Generalized Additive Models in Python. Zenodo. [DOI: 10.5281/zenodo.1208723](http://doi.org/10.5281/zenodo.1208723)

## Contact

To report an issue with BayesGAM, please use the [issue tracker](https://github.com/ouslan/BayesGAM/issues).

## License

GNU General Public License v3.0

## Getting Started

If you're new to BayesGAM, read [the Tour of BayesGAM](tour.md) for an introduction to the package.

## Indices and Tables

* [Genindex](genindex)
* [Modindex](modindex)
* [Search](search)
```