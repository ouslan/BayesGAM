"""
GAM toolkit
"""
from importlib.metadata import version, PackageNotFoundError

from bayesgam.bayesgam import GAM
from bayesgam.bayesgam import LinearGAM
from bayesgam.bayesgam import LogisticGAM
from bayesgam.bayesgam import GammaGAM
from bayesgam.bayesgam import PoissonGAM
from bayesgam.bayesgam import InvGaussGAM
from bayesgam.bayesgam import ExpectileGAM

from bayesgam.terms import l
from bayesgam.terms import s
from bayesgam.terms import f
from bayesgam.terms import te
from bayesgam.terms import intercept

__all__ = [
    'GAM',
    'LinearGAM',
    'LogisticGAM',
    'GammaGAM',
    'PoissonGAM',
    'InvGaussGAM',
    'ExpectileGAM',
    'l',
    's',
    'f',
    'te',
    'intercept',
]


__version__ = "0.0.0"  # placeholder for dynamic versioning
try:
    __version__ = version("bayesgam")
except PackageNotFoundError:
    # package is not installed
    pass
