import numpy as np

class GAM2():
    def __init__(self, formula:str, distribution:str="normal", link:str="normal"):
        self.formula = formula 
        self.distribution = distribution
        self.link = link
    
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
        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        mu = self.predict_mu(X)

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(
                weights, name='sample weights', ndim=1, verbose=self.verbose
            )
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('float64')

        return self._loglikelihood(y, mu, weights=weights)