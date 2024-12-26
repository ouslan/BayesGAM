from numpy.linalg import LinAlgError
import numpy as np
import scipy as sp
import warnings
import numbers

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

def check_y(y:np.ndarray, link:object, dist:str, min_samples:int=1, verbose:bool=True) -> np.ndarray:
    """ 
    tool to ensure that the targets:
    - are in the domain of the link function
    - are numerical
    - have at least min_samples
    - is finite

    Parameters
    ----------
    y : array-like
    link : Link object
    dist : Distribution object
    min_samples : int, default: 1
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    y : array containing validated y-data
    """
    y = np.ravel(y)

    y = check_array(
        y,
        force_2d=False,
        min_samples=min_samples,
        ndim=1,
        name="y data",
        verbose=verbose,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if np.any(np.isnan(link.link(y, dist))):
            raise ValueError(
                f"y data is not in domain of {link} link function. "
                f"Expected domain: {get_link_domain(link, dist)}, but found {[float("%.2f" % np.min(y)), float("%.2f" % np.max(y))]}"
            )
    return y

def check_X_y(X:np.ndarray, y:np.ndarray) -> None:
    """
    tool to ensure input and output data have the same number of samples

    Parameters
    ----------
    X : array-like
    y : array-like

    Returns
    -------
    None
    """
    if len(X) != len(y):
        raise ValueError(f"Inconsistent input and output data shapes. found X: {X.shape} and y: {y.shape}")

def check_lengths(*arrays) -> None:
    """
    tool to ensure input and output data have the same number of samples

    Parameters
    ----------
    *arrays : iterable of arrays to be checked

    Returns
    -------
    None
    """
    first_length = len(arrays[0])
    
    for i, array in enumerate(arrays[1:], 1):
        if len(array) != first_length:
            raise ValueError(f"Array at index {i} has a different length ({len(array)}). Expected length: {first_length}.")

def check_param(param, param_name:str, dtype, constraint=None, iterable:bool=True, max_depth:int=2):
    """
    checks the dtype of a parameter,
    and whether it satisfies a numerical contraint

    Parameters
    ---------
    param : object
    param_name : str, name of the parameter
    dtype : str, desired dtype of the parameter
    contraint : str, default: None
                numerical constraint of the parameter.
                if None, no constraint is enforced
    iterable : bool, default: True
               whether to allow iterable param
    max_depth : int, default: 2
                maximum nesting of the iterable.
                only used if iterable == True
    Returns
    -------
    list of validated and converted parameter(s)
    """
    msg = []
    msg.append(param_name + " must be " + dtype)
    if iterable:
        msg.append(
            " or nested iterable of depth "
            + str(max_depth)
            + " containing "
            + dtype
            + "s"
        )

    msg.append(", but found " + param_name + " = {}".format(repr(param)))

    if constraint is not None:
        msg = (" " + constraint).join(msg)
    else:
        msg = "".join(msg)

    # check param is numerical
    try:
        param_dt = np.array(
            flatten(param)
        )  # + np.zeros_like(flatten(param), dtype='int')
        # param_dt = np.array(param).astype(dtype)
    except (ValueError, TypeError):
        raise TypeError(msg)

    # check iterable
    if iterable:
        if check_iterable_depth(param) > max_depth:
            raise TypeError(msg)
    if (not iterable) and isiterable(param):
        raise TypeError(msg)

    # check param is correct dtype
    if not (param_dt == np.array(flatten(param)).astype(float)).all():
        raise TypeError(msg)

    # check constraint
    if constraint is not None:
        if not (eval("np." + repr(param_dt) + constraint)).all():
            raise ValueError(msg)

    return param

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
        return np.concatenate([np.min(data) - 0.5, np.max(data) + 0.5])
    else:
        knots = np.concatenate([np.min(data), np.max(data)])
        if knots[0] == knots[1] and verbose:
            warnings.warn(
                'Data contains constant feature. '
                'Consider removing and setting fit_intercept=True',
                stacklevel=2,
            )
        return knots

def b_spline_basis(
    x,
    edge_knots,
    n_splines=20,
    spline_order=3,
    sparse=True,
    periodic=True,
    verbose=True,
):
    """
    Generate B-spline basis functions using vectorized De Boor recursion.
    The basis functions extrapolate linearly past the end-knots.

    Parameters
    ----------
    x : array-like, 1D
        The data points at which to evaluate the B-spline basis functions.
    
    edge_knots : array-like
        Locations of the 2 edge knots (start and end values).
    
    n_splines : int, optional, default=20
        Number of splines to generate. Must be >= spline_order + 1.
    
    spline_order : int, optional, default=3
        Order of the spline basis to create.
    
    sparse : bool, optional, default=True
        Whether to return a sparse basis matrix or not.
    
    periodic : bool, optional, default=True
        Whether to repeat basis functions (True) or linearly extrapolate (False).
    
    verbose : bool, optional, default=True
        Whether to print warnings.

    Returns
    -------
    basis : sparse csc matrix or array
        B-spline basis functions with shape (len(x), n_splines).
    """
    
    # Ensure x is a 1D array
    x = np.ravel(x)
    if x.ndim != 1:
        raise ValueError(f"Data must be 1-D, but found {x.ndim} dimensions.")
    
    # Validate the number of splines and spline order
    if not isinstance(n_splines, numbers.Integral) or n_splines < 1:
        raise ValueError("n_splines must be an integer >= 1.")
    
    if not isinstance(spline_order, numbers.Integral) or spline_order < 0:
        raise ValueError("spline_order must be an integer >= 1.")
    
    if n_splines < spline_order + 1:
        raise ValueError(f"n_splines must be >= spline_order + 1. Found n_splines = {n_splines} and spline_order = {spline_order}.")

    if n_splines == 0 and verbose:
        warnings.warn("Requested 1 spline, which is equivalent to fitting an intercept.", stacklevel=2)

    # Adjust n_splines if periodic
    n_splines += spline_order * periodic

    # Rescale edge_knots to [0, 1] and generate boundary knots
    edge_knots = np.sort(edge_knots)
    offset = edge_knots[0]
    scale = edge_knots[-1] - edge_knots[0]

    if scale == 0: # Avoid division by zero
        scale = 1

    boundary_knots = np.linspace(0, 1, n_splines - spline_order + 1)
    diff = np.diff(boundary_knots[:2])[0]

    # Rescale x to [0, 1]
    x = (x - offset) / scale

    # Wrap x for periodic values
    if periodic:
        x = x % (1 + 1e-9)

    # Append 0 and 1 to handle boundary conditions
    x = np.r_[x, 0.0, 1.0]

    # Identify extrapolation indices
    x_extrapolte_l = x < 0
    x_extrapolte_r = x > 1
    x_interpolate = ~(x_extrapolte_r | x_extrapolte_l)

    # Ensure x is 2D for matrix operations
    x = np.atleast_2d(x).T

    # Augment knots for spline calculation
    aug = np.arange(1, spline_order + 1) * diff
    aug_knots = np.r_[-aug[::-1], boundary_knots, 1 + aug]
    aug_knots[-1] += 1e-9  # Ensure last knot is inclusive

    # Prepare the Haar basis (step function representation)
    bases = (x >= aug_knots[:-1]).astype(int) * (x < aug_knots[1:]).astype(int)
    bases[-1] = bases[-2][::-1]  # Force symmetry at boundaries

    # Perform De Boor recursion (vectorized)
    maxi = len(aug_knots) - 1
    for m in range(2, spline_order + 2):
        maxi -= 1

        # Store previous bases to use for extrapolation
        prev_bases = bases.copy()

        # Left sub-basis
        num = (x - aug_knots[:maxi]) * bases[:, :maxi]
        denom = aug_knots[m - 1 : maxi + m - 1] - aug_knots[:maxi]
        left = num / denom

        # Right sub-basis
        num = (aug_knots[m : maxi + m] - x) * bases[:, 1 : maxi + 1]
        denom = aug_knots[m : maxi + m] - aug_knots[1 : maxi + 1]
        right = num / denom

        # Combine left and right bases
        bases = left + right

    # Adjust for periodicity, if required
    if periodic and spline_order > 0:
        bases[:, :spline_order] = np.maximum(bases[:, :spline_order], bases[:, -spline_order:])
        bases = bases[:, :-spline_order]  # Remove extra splines used for the periodic domain

    # Extrapolate for values outside the [0, 1] interval
    if (any(x_extrapolte_r) or any(x_extrapolte_l)) and spline_order > 0:
        bases[~x_interpolate] = 0.0

        # Use prev_bases to handle extrapolation properly
        denom = aug_knots[spline_order:-1] - aug_knots[:-spline_order - 1]
        left = prev_bases[:, :-1] / denom

        denom = aug_knots[spline_order + 1:] - aug_knots[1:-spline_order]
        right = prev_bases[:, 1:] / denom

        grads = spline_order * (left - right)

        # Handle left extrapolation
        if any(x_extrapolte_l):
            val = grads[0] * x[x_extrapolte_l] + bases[-2]
            bases[x_extrapolte_l] = val

        # Handle right extrapolation
        if any(x_extrapolte_r):
            val = grads[1] * (x[x_extrapolte_r] - 1) + bases[-1]
            bases[x_extrapolte_r] = val


    # Remove the artificially appended values at the boundaries (0 and 1)
    bases = bases[:-2]

    # Return sparse matrix if requested
    if sparse:
        return sp.csc_matrix(bases)

    return bases

def check_iterable_depth(obj, max_depth=100):
    """find the maximum depth of nesting of the iterable

    Parameters
    ----------
    obj : iterable
    max_depth : int, default: 100
        maximum depth beyond which we stop counting

    Returns
    -------
    int
    """

    def find_iterables(obj):
        iterables = []
        for item in obj:
            if isiterable(item):
                iterables += list(item)
        return iterables

    depth = 0
    while (depth < max_depth) and isiterable(obj) and len(obj) > 0:
        depth += 1
        obj = find_iterables(obj)
    return depth

def ylogydu(y, u):
    """
    tool to give desired output for the limit as y -> 0, which is 0

    Parameters
    ----------
    y : array-like of len(n)
    u : array-like of len(n)

    Returns
    -------
    np.array len(n)
    """
    mask = np.atleast_1d(y) != 0.0
    out = np.zeros_like(u)
    out[mask] = y[mask] * np.log(y[mask] / u[mask])
    return out

def flatten(iterable):
    """convenience tool to flatten any nested iterable

    example:

        flatten([[[],[4]],[[[5,[6,7, []]]]]])
        >>> [4, 5, 6, 7]

        flatten('hello')
        >>> 'hello'

    Parameters
    ----------
    iterable

    Returns
    -------
    flattened object
    """
    if isiterable(iterable):
        flat = []
        for item in list(iterable):
            item = flatten(item)
            if not isiterable(item):
                item = [item]
            flat += item
        return flat
    else:
        return iterable

def isiterable(obj, reject_string=True):
    """convenience tool to detect if something is iterable.
    in python3, strings count as iterables to we have the option to exclude them

    Parameters:
    -----------
    obj : object to analyse
    reject_string : bool, whether to ignore strings

    Returns:
    --------
    bool, if the object is itereable.
    """

    iterable = hasattr(obj, "__len__")

    if reject_string:
        iterable = iterable and not isinstance(obj, str)

    return iterable

def tensor_product(a, b, reshape=True):
    """
    compute the tensor protuct of two matrices a and b

    if a is (n, m_a), b is (n, m_b),
    then the result is
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise

    Parameters
    ---------
    a : array-like of shape (n, m_a)

    b : array-like of shape (n, m_b)

    reshape : bool, default True
        whether to reshape the result to be 2-dimensional ie
        (n, m_a * m_b)
        or return a 3-dimensional tensor ie
        (n, m_a, m_b)

    Returns
    -------
    dense np.ndarray of shape
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise
    """
    if a.ndim == 2:
        raise ValueError(f"matrix a must be 2-dimensional, but found {a.ndim} dimensions")
    if b.ndim == 2:
        raise ValueError(f"matrix b must be 2-dimensional, but found {b.ndim} dimensions")

    na, ma = a.shape
    nb, mb = b.shape

    if na != nb:
        raise ValueError("both arguments must have the same number of samples")

    if sp.sparse.issparse(a):
        a = a.toarray()

    if sp.sparse.issparse(b):
        b = b.toarray()

    tensor = np.tensordot(a, b, axes=0)

    if reshape:
        return tensor.reshape(na, ma * mb)

    return tensor

def get_link_domain(link, dist):
    """
    tool to identify the domain of a given monotonic link function

    Parameters
    ----------
    link : Link object
    dist : Distribution object

    Returns
    -------
    domain : list of length 2, representing the interval of the domain.
    """
    domain = np.array([-np.inf, -1, 0, 1, np.inf])
    domain = domain[~np.isnan(link.link(domain, dist))]
    return [domain[0], domain[-1]]

class TablePrinter:
    """Print a list of dictionaries as a table."""

    def __init__(self, fmt, sep=" ", ul=None):
        """
        Initialize the TablePrinter with format specifications.

        :param fmt: List of tuples, each containing:
            - heading (str): Column label
            - key (str): Dictionary key to retrieve value to print
            - width (int): Column width in characters

        :param sep: String, separation between columns (default is a space)
        :param ul: String, character to underline column label, or None for no underlining
        """
        self.fmt = str(sep).join(
            "{lb}{0}:{1}{rb}".format(key, width, lb="{", rb="}")
            for heading, key, width in fmt
        )
        self.head = {key: heading for heading, key, width in fmt}
        self.ul = {key: str(ul) * width for heading, key, width in fmt} if ul else None
        self.width = {key: width for heading, key, width in fmt}

    def row(self, data):
        """
        Generate a formatted row for a given data dictionary.

        :param data: Dictionary containing the row data
        :return: Formatted row string
        """
        return self.fmt.format(
            **{k: str(data.get(k, ""))[:w] for k, w in self.width.items()}
        )

    def __call__(self, data_list):
        """
        Format a list of dictionaries as a table.

        :param data_list: List of dictionaries containing the table data
        :return: Formatted table as a string
        """
        formatted_rows = [self.row(data) for data in data_list]

        # Insert the header row at the top
        formatted_rows.insert(0, self.row(self.head))

        # Insert the underlining row, if applicable
        if self.ul:
            formatted_rows.insert(1, self.row(self.ul))

        return "\n".join(formatted_rows)