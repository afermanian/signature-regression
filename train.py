import iisignature as isig
import math
import numpy as np

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from tools import get_sigX
import matplotlib.pyplot as plt
import seaborn as sns
from skfda.representation.basis import VectorValued, BSpline, Fourier
from skfda.representation.grid import FDataGrid
from skfda.ml.regression import LinearRegression

sns.set()


class SignatureRegression(object):
    """ Signature regression class

    Parameters
    ----------
    d: int
        Dimension of the space of the paths X, from which an output Y is
        learned.

    k: int
            Truncation order of the signature

    scaling: boolean, default=True
        Whether to scale the predictor matrix to have zero mean and unit variance
    """

    def __init__(self, k, scaling=False, alpha=None):
        self.scaling = scaling
        self.reg = Ridge(normalize=False, fit_intercept=False, solver='svd')
        self.k = k
        self.alpha = alpha
        if self.scaling:
            self.scaler = StandardScaler()

    def fit(self, X, Y, alphas=np.linspace(10 ** (-6), 100, num=1000)):
        """Fit a signature ridge regression.

        Parameters
        ----------
        Y: array, shape (n)
            Array of target values.

        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing
            the coordinates in R^d of n piecewise linear paths, each composed of
            n_points.

        alpha: float, default=1
            Regularization parameter in the Ridge regression.

        Returns
        -------
        reg: object
            Instance of sklearn.linear_model.Ridge

        Ypred: array, shape (n)
            Array of predicted values.
        """

        sigX = get_sigX(X, self.k)
        if self.scaling:
            self.scaler.fit(sigX)
            sigX = self.scaler.transform(sigX)

        if self.alpha is not None:
            self.reg.alpha_ = self.alpha
        else:
            reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd')
            reg_cv.fit(sigX, Y)
            self.alpha = reg_cv.alpha_
            self.reg.alpha_ = self.alpha
        self.reg.fit(sigX, Y)
        return self.reg

    def predict(self, X):
        """Outputs prediction of self.reg, already trained with signatures
        truncated at order k.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing
            the coordinates in R^d of n piecewise linear paths, each composed of
            n_points.

        Returns
        -------
        Ypred: array, shape (n)
            Array of predicted values.
        """

        sigX = get_sigX(X, self.k)
        if self.scaling:
            sigX = self.scaler.transform(sigX)
        Ypred = self.reg.predict(sigX)
        return Ypred

    def get_loss(self, X, Y, plot=False):
        """Computes the minimum empirical squared loss obtained with a Ridge
        regression on signatures truncated at k.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing
            the coordinates in R^d of n piecewise linear paths, each composed of
            n_points.

        Y: array, shape (n)
            Array of target values.

        plot: boolean, default=False
            If True, plots the regression coefficients and a scatter plot of the
            target values Y against its predicted values Ypred to assess the
            quality
            of the fit.

        Returns
        -------
        hatL: float
            The squared loss, that is the sum of the squares of Y-Ypred, where
            Ypred are the fitted values of the Ridge regression of Y against
            signatures of X truncated at k.
        """
        Ypred = self.predict(X)
        if plot:
            plt.scatter(Y, Ypred)
            plt.plot([0.9 * np.min(Y), 1.1 * np.max(Y)], [0.9 * np.min(Y), 1.1 * np.max(Y)], '--', color='black')
            plt.title("Ypred against Y")
            plt.show()
        return np.mean((Y - Ypred) ** 2)


class SignatureOrderSelection(object):
    """Estimation of the order of truncation of the signature
    Parameters
    ----------
    d: int
        Dimension of the space of the paths X, from which an output Y is
        learned.

    Kpen: float, default=1
        Constant in front of the penalization, it has to be a positive
        number.

    rho: float
        Parameter of the penalization: power of 1/n. It should satisfy :
        0<rho<1/2.

    max_k: int,
        Maximal value of truncation order to consider. It should be large
        enough so that the function k -> hatL(k)+penalization(k) does not
        decrease anymore.
    """

    def __init__(self, d, rho=0.4, Kpen=None, alpha=None, max_features=None):
        self.d = d
        self.rho = rho
        self.Kpen = Kpen
        self.alpha = alpha
        if max_features is not None:
            self.max_features = max_features
        else:
            self.max_features = 10 ** 4
        self.max_k = math.floor((math.log(self.max_features * (d - 1) + 1) / math.log(d)) - 1)

    def fit_alpha(self, X, Y):
        sigreg = SignatureRegression(1)
        sigreg.fit(X, Y)
        self.alpha = sigreg.reg.alpha_
        return self.alpha

    def get_penalization(self, n, k, Kpen):
        """Returns the penalization function used in the estimator of the
        truncation order, that is,
        pen_n(k)=Kpen*sqrt((d^(k+1)-1)/(d-1))/n^rho.

        Parameters
        ----------
        n: int
            Number of samples.

        k: int
            Truncation order of the signature.

        Kpen: float, default=1
            Constant in front of the penalization, it has to be a positive
            number.

        Returns
        -------
        pen_n(k):float
            The penalization pen_n(k)
        """
        if k == 0:
            size_sig = 1
        else:
            size_sig = isig.siglength(self.d, k) + 1

        return Kpen * n ** (-self.rho) * math.sqrt(size_sig)

    def slope_heuristic(self, X, Y, Kpen_values):
        """Implements the slope heuristic to select a value for Kpen, the
        unknown constant in front of the penalization. To this end, hatm is
        computed for several values of Kpen, and these values are then plotted
        against Kpen.

        Parameters
        ----------
        Kpen_values: array, shape (n_K)
            An array of potential values for Kpen. It must be calibrated so that
            any value of hatm between 1 and max_k is obtained with values of
            Kpen in Kpen_values.

        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing
            the coordinates in R^d of n piecewise linear paths, each composed of
            n_points.

        Y: array, shape (n)
            Array of target values.

        Returns
        -------
        hatm: array, shape (n_K)
            The estimator hatm obtained for each value of Kpen in Kpen_values.
        """
        if self.alpha is None:
            self.fit_alpha(X, Y)
        hatm = np.zeros(len(Kpen_values))
        loss = np.zeros(self.max_k + 1)
        for j in range(self.max_k + 1):
            sigReg = SignatureRegression(j, alpha=self.alpha)
            sigReg.fit(X, Y)
            loss[j] = sigReg.get_loss(X, Y)

        for i in range(len(Kpen_values)):
            # print(i)
            pen = np.zeros(self.max_k + 1)
            for j in range(self.max_k + 1):
                pen[j] = self.get_penalization(Y.shape[0], j, Kpen_values[i])
            hatm[i] = np.argmin(loss + pen)

        # Plot
        fig, ax = plt.subplots()
        jump = 1
        for i in range(self.max_k + 1):
            if i in hatm:
                xmin = Kpen_values[hatm == i][0]
                xmax = Kpen_values[hatm == i][-1]
                ax.hlines(i, xmin, xmax, colors='b')
                if i != 0:
                    ax.vlines(xmax, i, i - jump, linestyles='dashed', colors='b')
                jump = 1
            else:
                jump += 1
        ax.set(xlabel=r'$K_{pen}$', ylabel=r'$\hat{m}$')
        plt.show()

        return hatm

    def get_hatm(self, X, Y, Kpen_values=np.linspace(10 ** (-5), 10 ** 2, num=200), plot=False):
        """Computes the estimator of the truncation order by minimizing the sum
        of hatL and the penalization, over values of k from 1 to max_k.

        Parameters
        ----------
        Y: array, shape (n)
            Array of target values.

        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing
            the coordinates in R^d of n piecewise linear paths, each composed of
            n_points.

        Kpen: float, default=1
            Constant in front of the penalization, it has to be a positive
            number.

        plot: boolean, default=False
            If True, plots the functions k->hatL(k), k->pen(k) and k-> hatL
            (k)+pen(k). The latter is minimized at hatm.

        Returns
        -------
        hatm: int
            The estimator of the truncation order

        objective: array, shape (max_k)
            The array of values of the objective function, minimized at hatm.
        """
        if not self.Kpen:
            self.slope_heuristic(X, Y, Kpen_values)
            Kpen = float(input("Enter slope heuristic constant Kpen: "))
        else:
            Kpen = self.Kpen
        objective = np.zeros(self.max_k + 1)
        loss = np.zeros(self.max_k + 1)
        pen = np.zeros(self.max_k + 1)
        for i in range(self.max_k + 1):
            sigReg = SignatureRegression(i, alpha=self.alpha)
            sigReg.fit(X, Y)
            loss[i] = sigReg.get_loss(X, Y)
            pen[i] = self.get_penalization(Y.shape[0], i, Kpen)
            objective[i] = loss[i] + pen[i]
        hatm = np.argmin(objective)

        if plot:
            plt.plot(np.arange(self.max_k + 1), loss, label="loss")
            plt.plot(np.arange(self.max_k + 1), pen, label="penalization")
            plt.plot(np.arange(self.max_k + 1), objective, label="sum")
            plt.legend()
            plt.show()
        return hatm


class BasisRegression(object):
    def __init__(self, nbasis, basis_type='bspline'):
        self.nbasis = nbasis
        self.reg = LinearRegression()
        self.basis_type = basis_type

    def data_to_basis(self, X):
        grid_points = np.linspace(0, 1, X.shape[1])
        fd = FDataGrid(X, grid_points)
        basis_vec = []
        for i in range(X.shape[2]):
            if self.basis_type == 'bspline':
                basis_vec.append(BSpline(n_basis=self.nbasis))
            elif self.basis_type == 'fourier':
                basis_vec.append(Fourier(n_basis=self.nbasis))
        basis = VectorValued(basis_vec)
        fd_basis = fd.to_basis(basis)
        return fd_basis

    def fit(self, X, Y):
        fd_basis = self.data_to_basis(X)
        self.reg.fit(fd_basis, Y)
        self.coef = self.reg.coef_
        return self.reg

    def predict(self, X):
        fd_basis = self.data_to_basis(X)
        return self.reg.predict(fd_basis)

    def get_loss(self, X, Y, plot=False):
        Ypred = self.predict(X)
        if plot:
            plt.scatter(Y, Ypred)
            plt.plot([0.9 * np.min(Y), 1.1 * np.max(Y)], [0.9 * np.min(Y), 1.1 * np.max(Y)], '--', color='black')
            plt.title("Ypred against Y")
            plt.show()
        return np.mean((Y - Ypred) ** 2)


def select_nbasis_cv(X, Y, basis_type, nbasis_grid=np.arange(10) + 4):
    score = []

    for nbasis in nbasis_grid:
        kf = KFold(n_splits=5)
        score_i = []
        for train, test in kf.split(X):
            reg = BasisRegression(nbasis, basis_type)
            reg.fit(X[train], Y[train])
            score_i += [np.mean((reg.predict(X[test]) - Y[test]) ** 2)]
        score += [np.mean(score_i)]
    return nbasis_grid[np.argmin(score)]


def select_hatm_cv(X, Y, max_k=None, scaling=False, plot=False):
    d = X.shape[2]
    max_features = 10 ** 4
    if max_k is None:
        max_k = math.floor((math.log(max_features * (d - 1) + 1) / math.log(d)) - 1)
    score = []
    for k in range(max_k+1):
        kf = KFold(n_splits=5)
        score_i = []
        for train, test in kf.split(X):
            reg = SignatureRegression(k, scaling=scaling)
            reg.fit(X[train], Y[train])
            score_i += [reg.get_loss(X[test], Y[test])]
        score += [np.mean(score_i)]
    if plot:
        plt.plot(np.arange(max_k+1), score)
        plt.show()
    return np.argmin(score)