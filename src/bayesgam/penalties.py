import scipy as sp
import numpy as np
import warnings

def sparse_diff(array:np.ndarray, n:int=1, axis:int=-1) -> sp.sparse.csc_matrix:
    """
    A sparse version of np.diff that computes differences recursively.

    Parameters
    ----------
    array : sparse array
        The input sparse array.
    n : int, default: 1
        The number of times to apply the differencing.
    axis : int, default: -1
        The axis along which the differences are computed.

    Returns
    -------
    diff_array : sparse array
        A sparse array of the same shape as the input, with the 'axis' dimension reduced
        by 'n'.

    Raises
    ------
    ValueError
        If 'n' is not a non-negative integer.
    """
    # Ensure n is a non-negative integer
    if n < 0 or not isinstance(n, int):
        raise ValueError(f"Expected non-negative integer for 'n', but got: {n}")

    # If array is not sparse, warn the user
    if not sp.sparse.issparse(array):
        warnings.warn("Array is not sparse. Consider using numpy.diff.")

    # Base case: if n is 0, return the original array
    if n == 0:
        return array

    slices = [slice(None)] * array.ndim
    slice1 = slices.copy()
    slice2 = slices.copy()

    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    diff_previous = sparse_diff(array, n - 1, axis=axis)

    return diff_previous[slice1] - diff_previous[slice2]

def derivative(n:int, derivative:int=2, periodic:bool=False) -> sp.sparse.csc_matrix:
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes the squared differences between basis coefficients.

    Parameters
    ----------
    n : int
        number of splines

    coef : unused
        for compatibility with constraints

    derivative: int, default: 2
        which derivative do we penalize.
        derivative is 1, we penalize 1st order derivatives,
        derivative is 2, we penalize 2nd order derivatives, etc

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n == 1:
        # no derivative for constant functions
        return sp.sparse.csc_matrix(0.0)
    D = sparse_diff(
        sp.sparse.identity(
            n + 2 * derivative * periodic).tocsc(), n=derivative
    ).tolil()

    if periodic:
        # wrap penalty
        cols = D[:, :derivative]
        D[:, -2 * derivative : -derivative] += cols * (-1) ** derivative

        # do symmetric operation on lower half of matrix
        n_rows = int((n + 2 * derivative) / 2)
        D[-n_rows:] = D[:n_rows][::-1, ::-1]

        # keep only the center of the augmented matrix
        D = D[derivative:-derivative, derivative:-derivative]
    return D.dot(D.T).tocsc()

def apply_periodic_penalty(penalty_func, n:int, coef:int, derivative:int=2):
    """
    Applies the penalty function for periodic features.

    Parameters
    ----------
    penalty_func : callable
        The function to calculate the penalty.
    n : int
        The number of terms.
    coef : array-like
        The coefficients used in the penalty calculation.
    derivative : int, default: 2
        The order of the derivative.

    Returns
    -------
    penalty_result : sparse matrix
        The penalty matrix considering periodicity.
    """
    return penalty_func(n, coef, derivative=derivative, periodic=True)

def periodic(n:np.ndarray, coef:np.ndarray, derivative:int=2, penalty_func=None):
    """
    Wraps a penalty function to calculate the penalty for periodic features.

    Parameters
    ----------
    n : int
        The number of terms in the penalty calculation.
    coef : array-like
        Coefficients for the penalty function.
    derivative : int, default: 2
        The order of the derivative to apply (default is 2).
    penalty_func : callable, default: None
        The penalty function to apply.

    Returns
    -------
    penalty_result : sparse matrix
        The computed penalty matrix considering periodicity.
    """
    if penalty_func is None:
        penalty_func = derivative  # Default penalty function if none is provided

    return apply_periodic_penalty(penalty_func, n, coef, derivative)

def l2(n:np.ndarray) -> np.ndarray:
    """
    Builds a penalty matrix for P-Splines with categorical features.
    Penalizes the squared value of each basis coefficient.

    Parameters
    ----------
    n : int
        number of splines

    coef : unused
        for compatibility with constraints

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return sp.sparse.eye(n).tocsc()

def monotonicity(n:int, coef:np.ndarray, increasing:bool=True) -> sp.sparse.csc_matrix:
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of monotonicity in the feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function
    increasing : bool, default: True
        whether to enforce monotic increasing, or decreasing functions
    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n != len(coef.ravel()):
        raise ValueError(f'dimension mismatch: expected n equals len(coef), but found n = {n}, coef.shape = {coef.shape}.')

    if n == 1:
        # no monotonic penalty for constant functions
        return sp.sparse.csc_matrix(0.0)

    if increasing:
        # only penalize the case where coef_i-1 > coef_i
        mask = sp.sparse.diags((np.diff(coef.ravel()) < 0).astype(float))
    else:
        # only penalize the case where coef_i-1 < coef_i
        mask = sp.sparse.diags((np.diff(coef.ravel()) > 0).astype(float))

    D = sparse_diff(sp.sparse.identity(n).tocsc(), n=1) * mask
    return D.dot(D.T).tocsc()

def monotonic_inc(n:int, coef:np.ndarray) -> sp.sparse.csc_matrix:
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of a monotonic increasing feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like, coefficients of the feature function

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return monotonicity(n, coef, increasing=True)

def convexity_(n, coef, convex=True):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of convexity in the feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function
    convex : bool, default: True
        whether to enforce convex, or concave functions
    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n != len(coef.ravel()):
        raise ValueError(
            f'dimension mismatch: expected n equals len(coef), but found n = {n}, coef.shape = {coef.shape}.')

    if n == 1:
        # no convex penalty for constant functions
        return sp.sparse.csc_matrix(0.0)

    if convex:
        mask = sp.sparse.diags((np.diff(coef.ravel(), n=2) < 0).astype(float))
    else:
        mask = sp.sparse.diags((np.diff(coef.ravel(), n=2) > 0).astype(float))

    D = sparse_diff(sp.sparse.identity(n).tocsc(), n=2) * mask
    return D.dot(D.T).tocsc()

def none(n:int) -> sp.sparse.csc_matrix:
    """
    Build a matrix of zeros for features that should go unpenalized

    Parameters
    ----------
    n : int
        number of splines
    coef : unused
        for compatibility with constraints

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return sp.sparse.csc_matrix(np.zeros((n, n)))

