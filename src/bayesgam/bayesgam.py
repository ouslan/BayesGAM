from bayesgam.distributions import NormalDist, BinomialDist, PoissonDist, GammaDist, InvGaussDist
from bayesgam.links import IdentityLink, LogitLink, LogLink, InverseLink, InvSquaredLink
from bayesgam.terms import TermList
from bayesgam.utils import check_y, check_X, check_array
import numpy as np

class GAM():
    def __init__(self, formula:str, distribution:str="normal", link:str="normal"):
        self.formula = formula 
        
        match distribution:
            case "normal":
                self.distribution = NormalDist
            case "binomial":
                self.distribution = BinomialDist
            case "poisson":
                self.distribution = PoissonDist
            case "gamma":
                self.distribution = GammaDist
            case "inv_gauss":
                self.distribution = InvGaussDist
            case _:
                raise NameError(f"The {distribution} is currently not supported.")

        match link:
            case "identity":
                self.link = IdentityLink
            case "logit":
                self.link = LogitLink
            case "inverse":
                self.link = InverseLink
            case "log":
                self.link = LogLink
            case "inverse-squared":
                self.link = InvSquaredLink
            case _: 
                raise NameError(f"The {link} link is currently not supported.")
    
    def s(self, x):
        return np.sin(x)
    
    def f(self, x):
        return x 
        
    def lin(self,x):
        return x

    def te(self,x):
        return x

    def intercept(self,x):
        return x
    
    def fit(self, X, Y):
        formula = self.formula
        
        local_scope = {
            's': self.s,
            'f': self.f,
            'l': self.lin,
            'te': self.te,
            'X': X,
            'intercept': self.intercept
        }
        
        result = eval(formula, {"__builtins__": None}, local_scope)

        return result

    def loglikelihood(self, X, y, weights=None):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing the input dataset
        y : array-like of shape (n,)
            containing target values
        weights : array-like of shape (n,), optional
            containing sample weights

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        y = check_y(y, self.link, self.distribution)
        mu = self.predict_mu(X)

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(
                weights, name='sample weights', ndim=1
            )
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('float64')

        return self.distribution.log_pdf(y=y, mu=mu, weights=weights).sum()

    def predict_mu(self, X):
        """
        preduct expected value of target given model and input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features),
            containing the input dataset

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing expected values under the model
        """

        X = check_X(
            X,
            n_feats=self.statistics_['m_features'],
            edge_knots=self.edge_knots_,
            dtypes=self.dtype,
            features=self.feature,
            verbose=self.verbose,
        )

        lp = self._linear_predictor(X)
        return self.link.mu(lp, self.distribution)

    def linear_predictor(self, X=None, modelmat=None, b=None, term=-1):
        """linear predictor
        compute the linear predictor portion of the model
        ie multiply the model matrix by the spline basis coefficients

        Parameters
        ---------
        at least 1 of (X, modelmat)
            and
        at least 1 of (b, feature)

        X : array-like of shape (n_samples, m_features) or None, optional
            containing the input dataset
            if None, will attempt to use modelmat

        modelmat : array-like or None, optional
            contains the spline basis for each feature evaluated at the input
            values for each feature, ie model matrix
            if None, will attempt to construct the model matrix from X

        b : array-like or None, optional
            contains the spline coefficients
            if None, will use current model coefficients

        feature : int, optional
                  feature for which to compute the linear prediction
                  if -1, will compute for all features

        Returns
        -------
        lp : np.array of shape (n_samples,)
        """
        if modelmat is None:
            modelmat = self.modelmat(X, term=term)
        if b is None:
            b = self.coef_[self.terms.get_coef_indices(term)]
        return modelmat.dot(b).flatten()
    
    def modelmat(self, X, term=-1):
        """
        Builds a model matrix, B, out of the spline basis for each feature

        B = [B_0, B_1, ..., B_p]

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing the input dataset
        term : int, optional
            term index for which to compute the model matrix
            if -1, will create the model matrix for all features

        Returns
        -------
        modelmat : sparse matrix of len n_samples
            containing model matrix of the spline basis for selected features
        """
        X = check_X(
            X,
            n_feats=self.statistics_['m_features'],
            edge_knots=self.edge_knots_,
            dtypes=self.dtype,
            features=self.feature,
            verbose=self.verbose,
        )

        return self.terms.build_columns(X, term=term)
    
    def summary(
        self,
        distribution:str,
        link_function:str, 
        num_samples:int, 
        log_likelihood:float, 
        aic:float, 
        aicc:float,
        gcv:float, 
        scale:float, 
        dof:float,
        pseudo_r_squared:float, 
        terms:list) -> str:
    
        output = []
        output.append("LinearGAM")
        output.append(f"{'='*45} {'='*45}")
        output.append(f"Distribution: {distribution:>31} Effective DoF: {dof:>30}")
        output.append(f"Link Function: {link_function:>30} Log Likelihood: {log_likelihood:>29}")
        output.append(f"Number of Samples: {num_samples:>26} AIC: {aic:>40}")
        output.append(f"{'':>45} AICc: {aicc:>39}")
        output.append(f"{'':>45} GCV: {gcv:>40}")
        output.append(f"{'':>45} Scale: {scale:>38}")
        output.append(f"{'':>45} Pseudo R-Squared: {pseudo_r_squared:>27}")
        output.append(f"{'='*45} {'='*45}")
        
        # Add feature functions and details header
        output.append(f"{'Feature Function':<30}{'Lambda':<16}{'Rank':<10}{'EDoF':<10}{'P > x':<15}{'Sig. Code':>10}")
        output.append(f"{'='*29} {'='*15} {'='*9} {'='*9} {'='*16} {'='*8}") 
        
        # Iterate over the terms and add their details to the output
        for term in terms:
            feature_name = term[0]
            lambda_value = term[1]
            rank = term[2]
            edof = term[3]
            p_value = term[4]
            sig_code = '*' if p_value < 0.01 else ''
            
            output.append(f"{feature_name:<30}{lambda_value:<16}{rank:<10}{edof:<10}{p_value:<15}{sig_code:>10}")
        
        output.append(f"{'='*45} {'='*45}")
        output.append("Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

        # Display Important warnings
        output.append("\n\033[1;33mWARNING:\033[0m Fitting splines and a linear function to a feature introduces a model\n"
                    f"{' '*8} identifiability problem.")
        output.append("\033[1;33mWARNING:\033[0m p-values calculated in this manner behave correctly for un-penalized models\n"
                    f"{' '*8} or models with known smoothing parameters")
        output.append("but when smoothing parameters have been estimated, the p-values are typically lower than they should be, meaning that the tests reject the null too readily.")

        return "\n".join(output)
        