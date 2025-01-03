from bayesgam.utils import ylogydu
import scipy as sp
import numpy as np


class Distribution():

    def phi(self, y, mu, edof, weights):
        """
        GLM scale parameter.
        for Binomial and Poisson families this is unity
        for Normal family this is variance

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        edof : float
            estimated degrees of freedom
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        scale : estimated model scale
        """
        return np.sum(weights * self.V(mu) ** -1 * (y - mu) ** 2) / (len(mu) - edof)

    def V(self, mu):
        """
        glm Variance function.

        if
            Y ~ ExpFam(theta, scale=phi)
        such that
            E[Y] = mu = b'(theta)
        and
            Var[Y] = b''(theta) * phi / w

        then we seek V(mu) such that we can represent Var[y] as a fn of mu:
            Var[Y] = V(mu) * phi

        ie
            V(mu) = b''(theta) / w

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        V(mu) : np.array of length n
        """
        return np.ones_like(mu)

class NormalDist(Distribution):

    def __init__(self, scale=None) -> None:
        self.scale = scale

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        scale = self.scale / weights
        return sp.stats.norm.logpdf(y, loc=mu, scale=scale)

    def V(self, mu):
        """
        glm Variance function.

        if
            Y ~ ExpFam(theta, scale=phi)
        such that
            E[Y] = mu = b'(theta)
        and
            Var[Y] = b''(theta) * phi / w

        then we seek V(mu) such that we can represent Var[y] as a fn of mu:
            Var[Y] = V(mu) * phi

        ie
            V(mu) = b''(theta) / w

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        V(mu) : np.array of length n
        """
        return np.ones_like(mu)

    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a gaussian linear model, this is equal to the SSE

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled

        Returns
        -------
        deviances : np.array of length n
        """
        dev = (y - mu) ** 2
        if scaled:
            dev /= self.scale
        return dev

class BinomialDist():
    pass

class PoissonDist():
    pass

class GammaDist():
    pass

class InvGaussDist():
    pass