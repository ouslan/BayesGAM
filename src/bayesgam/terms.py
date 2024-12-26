from bayesgam.utils import (
    isiterable,
    check_param,
    flatten,
    gen_edge_knots,
    b_spline_basis,
    tensor_product,
)
import numpy as np
import scipy as sp

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

        if not (hasattr(self, '_name')):
            self._name = 'term'

        super(Term, self).__init__(name=self._name)
        self._validate_arguments()

    def build_penalties(self):
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