import iisignature as isig
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import seaborn as sns
from skfda.representation.basis import VectorValued, BSpline, Fourier
from skfda.representation.grid import FDataGrid
from skfda.ml.regression import LinearRegression
from skfda.preprocessing.dim_reduction.projection import FPCA
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tools import get_sigX

sns.set()
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


class SignatureRegression(object):
    """ Signature regression class

    Parameters
    ----------
    k: int
            Truncation order of the signature

    scaling: boolean, default=True
        Whether to scale the predictor matrix to have zero mean and unit variance

    alpha: float, default=None
        Regularization parameter in the Ridge regression

    Attributes
    ----------
    reg: object
        Instance of sklearn.linear_model.Ridge

    scaler: object
        Instance of sklearn.preprocessing.StandardScaler
    """

    def __init__(self, k, scaling=False, alpha=None):
        self.scaling = scaling
        self.reg = Ridge(normalize=False, fit_intercept=False, solver='svd')
        self.k = k
        self.alpha = alpha
        if self.scaling:
            self.scaler = StandardScaler()

    def fit(self, X, Y, alphas=np.linspace(10 ** (-5), 100, num=100)):
        """Fit a signature ridge regression.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

         Y: array, shape (n)
            Array of target values.

        alphas: array, default=np.linspace(10 ** (-6), 100, num=1000)
            Grid for the cross validation search of the regularization parameter in the Ridge regression.

        Returns
        -------
        reg: object
            Instance of sklearn.linear_model.Ridge
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
        """Outputs prediction of self.reg, already trained with signatures truncated at order k.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

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
        """Computes the empirical squared loss obtained with a Ridge regression on signatures truncated at k.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        plot: boolean, default=False
            If True, plots the regression coefficients and a scatter plot of the target values Y against its predicted
            values Ypred to assess the quality of the fit.

        Returns
        -------
        hatL: float
            The squared loss, that is the sum of the squares of Y-Ypred, where Ypred are the fitted values of the Ridge
            regression of Y against signatures of X truncated at k.
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
        Dimension of the space of the paths X, from which an output Y is learned.

    rho: float
        Parameter of the penalization: power of 1/n. It should satisfy : 0<rho<1/2.

    Kpen: float, default=none
        Constant in front of the penalization, it has to be a positive number.

    alpha: float, default=None
        Regularization parameter in the Ridge regression.

    max_features: int,
        Maximal size of coefficients considered.

    Attributes
    ----------
    max_k: int,
        Maximal value of signature truncation to keep the number of features below max_features.

    """
    def __init__(self, d, rho=0.4, Kpen=None, alpha=None, max_features=10 ** 3):
        self.d = d
        self.rho = rho
        self.Kpen = Kpen
        self.alpha = alpha

        self.max_features = max_features
        self.max_k = math.floor((math.log(self.max_features * (d - 1) + 1) / math.log(d)) - 1)

    def fit_alpha(self, X, Y):
        """ Find alpha by cross validation with signatures truncated at order 1.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        Returns
        -------
            alpha: float
                Regularization parameter
        """
        sigreg = SignatureRegression(1)
        sigreg.fit(X, Y)
        self.alpha = sigreg.reg.alpha_
        return self.alpha

    def get_penalization(self, n, k, Kpen):
        """Returns the penalization function used in the estimator of the truncation order, that is,
        pen_n(k)=Kpen*sqrt((d^(k+1)-1)/(d-1))/n^rho.

        Parameters
        ----------
        n: int
            Number of samples.

        k: int
            Truncation order of the signature.

        Kpen: float, default=1
            Constant in front of the penalization, it has to be strictly positive.

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

    def slope_heuristic(self, X, Y, Kpen_values, savefig=False):
        """Implements the slope heuristic to select a value for Kpen, the
        unknown constant in front of the penalization. To this end, hatm is
        computed for several values of Kpen, and these values are then plotted
        against Kpen.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        Kpen_values: array, shape (n_K)
            An array of potential values for Kpen. It must be calibrated so that any value of hatm between 1 and max_k
            is obtained with values of Kpen in Kpen_values.

        savefig: boolean, default=False
            If True, saves the slope heuristics plot of hatm against Kpen.

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
        palette = sns.color_palette('colorblind')
        fig, ax = plt.subplots()
        jump = 1
        for i in range(self.max_k + 1):
            if i in hatm:
                xmin = Kpen_values[hatm == i][0]
                xmax = Kpen_values[hatm == i][-1]
                ax.hlines(i, xmin, xmax, colors='b')
                if i != 0:
                    ax.vlines(xmax, i, i - jump, linestyles='dashed', colors=palette[0])
                jump = 1
            else:
                jump += 1
        ax.set(xlabel=r'$K_{pen}$', ylabel=r'$\hat{m}$')
        if savefig:
            plt.savefig('Figures/kpen_selection.pdf', bbox_inches='tight')
        plt.show()

        return hatm

    def get_hatm(self, X, Y, Kpen_values=np.linspace(10 ** (-5), 10 ** 2, num=200), plot=False, savefig=False):
        """Computes the estimator of the truncation order by minimizing the sumof hatL and the penalization, over
        values of k from 1 to max_k.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        Kpen_values: array, shape (n_K)
            An array of potential values for Kpen. It must be calibrated so that any value of hatm between 1 and max_k
            is obtained with values of Kpen in Kpen_values.

        plot: boolean, default=False
            If True, plots the functions k->hatL(k), k->pen(k) and k-> hatL
            (k)+pen(k). The latter is minimized at hatm.

        savefig: boolean, default=False
            If True, saves the slope heuristics plot of hatm against Kpen.

        Returns
        -------
        hatm: int
            The estimator of the truncation order
        """
        if not self.Kpen:
            self.slope_heuristic(X, Y, Kpen_values, savefig=savefig)
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
            palette = sns.color_palette('colorblind')
            plt.plot(np.arange(self.max_k + 1), loss, label=r"$\widehat{L}_n(m)$", color=palette[0])
            plt.plot(np.arange(self.max_k + 1), pen, label=r"$pen_n(m)$", color=palette[1], linestyle='dashed')
            plt.plot(np.arange(self.max_k + 1), objective, label=r"$\widehat{L}_n(m) + pen_n(m)$",
                     color=palette[2], linestyle='dotted')
            plt.legend()
            plt.xlabel(r'$m$')
        return hatm


class BasisRegression(object):
    """Class implementing functional linear models with basis functions for vector-valued covariates.

    Parameters
    ----------
    nbasis: int
        Number of basis functions.

    basis_type: str, default='bspline'
        Type of basis used, possible values are 'bspline', 'fourier' and 'fPCA'

    Attributes
    ----------
    reg: object
        Instance of skfda.ml.regression.LinearRegression

    coef: array, default=None
        Regression coefficients

    fpca_basis: object
        If basis_type='fPCA', instance of skfda.preprocessing.dim_reduction.projection.FPCA().
    """
    def __init__(self, nbasis, basis_type='bspline'):
        self.nbasis = nbasis
        self.reg = LinearRegression()
        self.basis_type = basis_type
        self.coef = None
        if self.basis_type=='fPCA':
            self.fpca_basis = FPCA(self.nbasis)

    def data_to_basis(self, X, fit_fPCA=True):
        """Project the data to basis functions.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise linear paths,
            each composed of n_points.

        fit_fPCA: boolean, default=True
            If n_basis='fPCA' and fit_fPCA=True, the basis functions are fitted to be the functional principal
            components of X.

        Returns
        -------
        fd_basis: object
            Instance of skfda.representation.basis.FDataBasis, the basis representation of X, where the type of basis is
            determined by self.n_basis.
        """
        grid_points = np.linspace(0, 1, X.shape[1])
        fd = FDataGrid(X, grid_points)
        basis_vec = []
        for i in range(X.shape[2]):
            if self.basis_type == 'bspline':
                basis_vec.append(BSpline(n_basis=self.nbasis))
            elif self.basis_type == 'fourier':
                basis_vec.append(Fourier(n_basis=self.nbasis))
            elif self.basis_type == 'fPCA':
                basis_vec.append(BSpline(n_basis=7))

        basis = VectorValued(basis_vec)
        fd_basis = fd.to_basis(basis)
        if self.basis_type == 'fPCA':
            if fit_fPCA:
                self.fpca_basis = self.fpca_basis.fit(fd_basis)
            fd_basis = self.fpca_basis.transform(fd_basis)
        return fd_basis

    def fit(self, X, Y):
        """Fit the functional linear model to X and Y

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        Returns
        -------
        reg: object
            Instance of skfda.ml.regression.LinearRegression
        """
        fd_basis = self.data_to_basis(X)
        self.reg.fit(fd_basis, Y)
        self.coef = self.reg.coef_
        return self.reg

    def predict(self, X):
        """Predict the output of self.reg for X.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Returns
        -------
        Ypred: array, shape (n)
            Array of predicted values.
        """
        fd_basis = self.data_to_basis(X, fit_fPCA=False)
        return self.reg.predict(fd_basis)

    def get_loss(self, X, Y, plot=False):
        """Computes the empirical squared loss obtained with the functional linear model

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        plot: boolean, default=False
            If True, plots the regression coefficients and a scatter plot of the target values Y against its predicted
            values Ypred to assess the quality of the fit.

        Returns
        -------
        hatL: float
            The squared loss, that is the sum of the squares of Y-Ypred, where Ypred are the fitted values of the Ridge
            regression of Y against signatures of X truncated at k.
        """
        Ypred = self.predict(X)
        if plot:
            plt.scatter(Y, Ypred)
            plt.plot([0.9 * np.min(Y), 1.1 * np.max(Y)], [0.9 * np.min(Y), 1.1 * np.max(Y)], '--', color='black')
            plt.title("Ypred against Y")
            plt.show()
        return np.mean((Y - Ypred) ** 2)


def select_nbasis_cv(X, Y, basis_type):
    """Select the optimal number of basis functions for a linear functional model implemented in the class
    BasisRegression by cross validation.

    Parameters
    ----------
    X: array, shape (n,n_points,d)
        Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
        linear paths, each composed of n_points.

    Y: array, shape (n)
        Array of target values.

    basis_type: str, default='bspline'
        Type of basis used, possible values are 'bspline', 'fourier' and 'fPCA'

    Returns
    -------
    nbasis: int
        Optimal number of basis functions.
    """
    score = []
    if basis_type == 'fPCA':
        nbasis_grid = np.arange(5) + 1
    else:
        nbasis_grid = np.arange(10) + 4
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
    """Select the optimal value of hatm for the signature linear model implemented in the class SignatureRegression by
    cross validation.

    Parameters
    ----------
    X: array, shape (n,n_points,d)
        Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
        linear paths, each composed of n_points.

    Y: array, shape (n)
        Array of target values.

    max_k: int,
        Maximal value of signature truncation to keep the number of features below max_features.

    scaling: boolean, default=False
        Whether to scale the predictor matrix to have zero mean and unit variance

    plot: boolean, default=False
        If true, plot the cross validation loss as a function of the truncation order.

    Returns
    -------
    hatm: int
        Optimal value of hatm.
    """
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