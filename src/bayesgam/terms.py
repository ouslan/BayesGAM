import numpy as np
import scipy as sp



def intercept(X):
    return sp.sparse.csc_matrix(np.ones((len(X), 1)))

class Term():

    def __init__(
        self,
        feature,
        lam=0.6,
        dtype='numerical',
        fit_linear=False,
        fit_splines=True,
        penalties='auto',
        constraints=None,
        verbose=False,
    ):
        """creates an instance of a Term

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        constraints : {None, 'convex', 'concave', 'monotonic_inc', 'monotonic_dec'}
            or callable or iterable

            Type of constraint to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.

        dtype : {'numerical', 'categorical'}
            String describing the data-type of the feature.

        fit_linear : bool
            whether to fit a linear model of the feature

        fit_splines : bool
            whether to fit spliens to the feature

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.feature = feature

        self.lam = lam
        self.dtype = dtype
        self.fit_linear = fit_linear
        self.fit_splines = fit_splines
        self.penalties = penalties
        self.constraints = constraints
        self.verbose = verbose
        self._validate_arguments()

    def _validate_arguments(self):
        """method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        # dtype
        if self.dtype not in ['numerical', 'categorical']:
            raise ValueError(
                "dtype must be in ['numerical','categorical'], "
                "but found dtype = {}".format(self.dtype)
            )

        # fit_linear XOR fit_splines
        if self.fit_linear == self.fit_splines:
            raise ValueError(
                'term must have fit_linear XOR fit_splines, but found: '
                'fit_linear= {}, fit_splines={}'.format(
                    self.fit_linear, self.fit_splines
                )
            )

        # penalties
        if not isiterable(self.penalties):
            self.penalties = [self.penalties]

        for i, p in enumerate(self.penalties):
            if not (hasattr(p, '__call__') or (p in PENALTIES) or (p is None)):
                raise ValueError(
                    "penalties must be callable or in "
                    "{}, but found {} for {}th penalty".format(
                        list(PENALTIES.keys()), p, i
                    )
                )

        # check lams and distribute to penalites
        if not isiterable(self.lam):
            self.lam = [self.lam]

        for lam in self.lam:
            check_param(lam, param_name='lam', dtype='float', constraint='>= 0')

        if len(self.lam) == 1:
            self.lam = self.lam * len(self.penalties)

        if len(self.lam) != len(self.penalties):
            raise ValueError(
                'expected 1 lam per penalty, but found '
                'lam = {}, penalties = {}'.format(self.lam, self.penalties)
            )

        # constraints
        if not isiterable(self.constraints):
            self.constraints = [self.constraints]

        for i, c in enumerate(self.constraints):
            if not (hasattr(c, '__call__') or (c in CONSTRAINTS) or (c is None)):
                raise ValueError(
                    "constraints must be callable or in "
                    "{}, but found {} for {}th constraint".format(
                        list(CONSTRAINTS.keys()), c, i
                    )
                )

        return self

    @property
    def istensor(self):
        return isinstance(self, TensorTerm)

    @property
    def isintercept(self):
        return isinstance(self, Intercept)

    @property
    def info(self):
        """get information about this term

        Parameters
        ----------

        Returns
        -------
        dict containing information to duplicate this term
        """
        info = self.get_params()
        info.update({'term_type': self._name})
        return info

    @classmethod
    def build_from_info(cls, info):
        """build a Term instance from a dict

        Parameters
        ----------
        cls : class

        info : dict
            contains all information needed to build the term

        Return
        ------
        Term instance
        """
        info = deepcopy(info)
        if 'term_type' in info:
            cls_ = TERMS[info.pop('term_type')]

            if issubclass(cls_, MetaTermMixin):
                return cls_.build_from_info(info)
        else:
            cls_ = cls
        return cls_(**info)

    @property
    def hasconstraint(self):
        """bool, whether the term has any constraints"""
        return np.not_equal(np.atleast_1d(self.constraints), None).any()

    @property
    @abstractproperty
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model"""
        pass

    @abstractmethod
    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        return self

    @abstractmethod
    def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        pass

    def build_penalties(self, verbose=False):
        """
        builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]


        Parameters
        ---------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form
        """
        if self.isintercept:
            return np.array([[0.0]])

        Ps = []
        for penalty, lam in zip(self.penalties, self.lam):
            if penalty == 'auto':
                if self.dtype == 'numerical':
                    if self._name == 'spline_term':
                        if self.basis in ['cp']:
                            penalty = 'periodic'
                        else:
                            penalty = 'derivative'
                    else:
                        penalty = 'l2'
                if self.dtype == 'categorical':
                    penalty = 'l2'
            if penalty is None:
                penalty = 'none'
            if penalty in PENALTIES:
                penalty = PENALTIES[penalty]

            P = penalty(self.n_coefs, coef=None)  # penalties dont need coef
            Ps.append(np.multiply(P, lam))
        return np.sum(Ps)

    def build_constraints(self, coef, constraint_lam, constraint_l2):
        """
        builds the GAM block-diagonal constraint matrix in quadratic form
        out of constraint matrices specified for each feature.

        behaves like a penalty, but with a very large lambda value, ie 1e6.

        Parameters
        ---------
        coefs : array-like containing the coefficients of a term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        constraint_l2 : float,
            loading to improve the numerical conditioning of the constraint
            matrix.

            typically this is a very small number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form
        """
        if self.isintercept:
            return np.array([[0.0]])

        Cs = []
        for constraint in self.constraints:
            if constraint is None:
                constraint = 'none'
            if constraint in CONSTRAINTS:
                constraint = CONSTRAINTS[constraint]

            C = constraint(self.n_coefs, coef) * constraint_lam
            Cs.append(C)

        Cs = np.sum(Cs)

        # improve condition
        if Cs.nnz > 0:
            Cs += sp.sparse.diags(constraint_l2 * np.ones(Cs.shape[0]))

        return Cs




class MetaTermMixin(object):
    _plural = [
        'feature',
        'dtype',
        'fit_linear',
        'fit_splines',
        'lam',
        'n_splines',
        'spline_order',
        'constraints',
        'penalties',
        'basis',
        'edge_knots_',
    ]
    _term_location = '_terms'

    def _super_get(self, name):
        return super(MetaTermMixin, self).__getattribute__(name)

    def _super_has(self, name):
        try:
            self._super_get(name)
            return True
        except AttributeError:
            return False

    def _has_terms(self):
        """bool, whether the instance has any sub-terms"""
        loc = self._super_get('_term_location')
        return (
            self._super_has(loc)
            and isiterable(self._super_get(loc))
            and len(self._super_get(loc)) > 0
            and all([isinstance(term, Term) for term in self._super_get(loc)])
        )

    def _get_terms(self):
        """get the terms in the instance

        Parameters
        ----------
        None

        Returns
        -------
        list containing terms
        """
        if self._has_terms():
            return getattr(self, self._term_location)

    def __setattr__(self, name, value):
        if self._has_terms() and name in self._super_get('_plural'):
            # get the total number of arguments
            size = np.atleast_1d(flatten(getattr(self, name))).size

            # check shapes
            if isiterable(value):
                value = flatten(value)
                if len(value) != size:
                    raise ValueError(
                        'Expected {} to have length {}, but found {} = {}'.format(
                            name, size, name, value
                        )
                    )
            else:
                value = [value] * size

            # now set each term's sequence of arguments
            for term in self._get_terms()[::-1]:
                # skip intercept
                if term.isintercept:
                    continue

                # how many values does this term get?
                n = np.atleast_1d(getattr(term, name)).size

                # get the next n values and set them on this term
                vals = [value.pop() for _ in range(n)][::-1]
                setattr(term, name, vals[0] if n == 1 else vals)

                term._validate_arguments()

            return
        super(MetaTermMixin, self).__setattr__(name, value)

    def __getattr__(self, name):
        if self._has_terms() and name in self._super_get('_plural'):
            # collect value from each term
            values = []
            for term in self._get_terms():
                # skip the intercept
                if term.isintercept:
                    continue

                values.append(getattr(term, name, None))
            return values

        return self._super_get(name)