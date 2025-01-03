from bayesgam.core import Core
import numpy as np



class IdentityLink():
    def __init__(self):
        """
        creates an instance of an IdentityLink object

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        self.name = "identity"

    def mu(self, lp):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu

        Parameters
        ----------
        lp : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        return lp

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return np.ones_like(mu)

class LogitLink():
    def __init__(self):
        """
        creates an instance of a LogitLink object

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        super(LogitLink, self).__init__(name='logit')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return np.log(mu) - np.log(dist.levels - mu)

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu

        Parameters
        ----------
        lp : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        elp = np.exp(np.clip(lp, -700, 700))
        return dist.levels * elp / (elp + 1)

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return dist.levels / (mu * (dist.levels - mu))

class LogLink():
    def __init__(self):
        """
        creates an instance of a LogitLink object

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        super(LogLink, self).__init__(name='log')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return np.log(mu)

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu

        Parameters
        ----------
        lp : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        return np.exp(np.clip(lp, -700, 700))     

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return 1.0 / mu

class InverseLink():
    def __init__(self):
        """
        creates an instance of a InverseLink object

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        super(InverseLink, self).__init__(name='inverse')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return mu**-1.0

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu

        Parameters
        ----------
        lp : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        return lp**-1.0

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return -1 * mu**-2.0

class InvSquaredLink():
    def __init__(self):
        """
        creates an instance of an InverseLink object

        Parameters
        ----------
        name : str, default: None

        Returns
        -------
        self
        """
        super(InvSquaredLink, self).__init__(name='inv_squared')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return mu**-2.0

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu

        Parameters
        ----------
        lp : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        return lp**-0.5

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return -2 * mu**-3.0
