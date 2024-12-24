from numpy.linalg import LinAlgError
import numpy as np
import scipy as sp
import sys
import warnings

def cholesky(A:np.ndarray) -> np.ndarray:
    """
    Choose the best possible cholesky factorizor.

    if possible, import the Scikit-Sparse sparse Cholesky method.
    Permutes the output L to ensure A = L.H . L

    otherwise defaults to numpy's non-sparse version

    Parameters
    ----------
    A : array-like
        array to decompose
    sparse : boolean, default: True
        whether to return a sparse array
    verbose : bool, default: True
        whether to print warnings
    """

    try:
        L = sp.linalg.cholesky(A, lower=False)
    except LinAlgError:
        raise ValueError("A must be a positive definite matrix")
    return L

def make_2d(array:np.ndarray, verbose=True) -> np.ndarray:
    """
    tiny tool to expand 1D arrays the way i want

    Parameters
    ----------
    array : array-like

    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    np.array of with ndim = 2
    """
    array = np.asarray(array)
    if array.ndim < 2:
        msg = f"Expected 2D input data array, but found {array.ndim}D. " "Expanding to 2D."
        if verbose:
            warnings.warn(msg)
        array = np.atleast_1d(array)[:, None]
    return array 

def check_array(
    array:np.ndarray, 
    force_2d:bool=False, 
    n_feats=None, 
    ndim=None, 
    min_samples:int=1, 
    name:str="Input data", 
    verbose:bool=True,
    ) -> np.ndarray:
    """
    tool to perform basic data validation.
    called by check_X and check_y.

    ensures that data:
    - is ndim dimensional
    - contains float-compatible data-types
    - has at least min_samples
    - has n_feats
    - is finite

    Parameters
    ----------
    array : array-like
    force_2d : boolean, default: False
        whether to force a 2d array. Setting to True forces ndim = 2
    n_feats : int, default: None
              represents number of features that the array should have.
              not enforced if n_feats is None.
    ndim : int default: None
        number of dimensions expected in the array
    min_samples : int, default: 1
    name : str, default: 'Input data'
        name to use when referring to the array
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    array : validated array
    """
    # make array
    if force_2d:
        array = make_2d(array, verbose=verbose)
        ndim = 2
    else:
        array = np.array(array)

    # cast to float
    dtype = array.dtype
    if dtype.kind not in ["i", "f"]:
        try:
            array = array.astype("float")
        except ValueError:
            raise ValueError(
                f"{name} must be type int or float, "
                f"but found type: {dtype.type}\n"
                "Try transforming data with a LabelEncoder first."
            )

    # check finite
    if not (np.isfinite(array).all()):
        raise ValueError(f"{name} must not contain Inf nor NaN")

    # check ndim
    if ndim is not None:
        if array.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions. found shape {array.shape}")

    # check n_feats
    if n_feats is not None:
        m = array.shape[1]
        if m != n_feats:
            raise ValueError(f"{name} must have {n_feats} features, but found {m}")

    # minimum samples
    n = array.shape[0]
    if n < min_samples:
        raise ValueError(f"{name} should have at least {min_samples} samples, but found {n}")

    return array

def sig_code(p_value:float) -> str:
    """create a significance code in the style of R's lm

    Arguments
    ---------
    p_value : float on [0, 1]

    Returns
    -------
    str
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '.'
    elif p_value >= 0.1 and p_value <= 1:
        return' '
    else:
        raise ValueError('p_value must be on [0, 1]')

def gen_edge_knots(data:np.ndarray, dtype:str, verbose:bool=True) -> np.ndarray:
    """
    generate uniform knots from data including the edges of the data

    for discrete data, assumes k categories in [0, k-1] interval

    Parameters
    ----------
    data : array-like with one dimension
    dtype : str in {'categorical', 'numerical'}
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    np.array containing ordered knots
    """
    if dtype not in ['categorical', 'numerical']:
        raise ValueError(f'unsupported dtype: {dtype}')
    if dtype == 'categorical':
        return np.r_[np.min(data) - 0.5, np.max(data) + 0.5]
    else:
        knots = np.concatenate([np.min(data), np.max(data)])
        if knots[0] == knots[1] and verbose:
            warnings.warn(
                'Data contains constant feature. '
                'Consider removing and setting fit_intercept=True',
                stacklevel=2,
            )
        return knots
