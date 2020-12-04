import math
import iisignature as isig
import numpy as np


def get_signature(path, order, norm_path=True):
    """ Returns the signature of a path truncated at a certain order.

    Parameters
    ----------
    path: array, shape (n_points,d)
        The array storing the n_points coordinates in R^d that constitute a
        piecewise linear path.

    order: int
        The truncation order of the signature

    norm_path: boolean, default=True
        Whether to normalise the path before computing the signature, such that
        the signature coefficients of order k are of order the length of the
        path.

    Returns
    -------
    sig: array, shape (p)
        Array containing the truncated signature coefficients of path. It is of
        shape p=(d^(order+1)-1)/(d-1)
    """
    if norm_path:
        path = path * (math.factorial(order)) ** (1 / order)
    return isig.sig(path, order)


def get_sigX(X, k, norm_path=False):
    """Returns a matrix containing signatures truncated at k of n samples
    given in the input tensor X.

    Parameters
    ----------
    X: array, shape (n,n_points,d)
        A 3-dimensional array, containing the coordinates in R^d of n
        piecewise linear paths, each composed of n_points.

    k: int
        Truncation order of the signature

    norm_path: boolean, default=True
        Whether to normalise the path before computing the signature, such that
        the signature coefficients of order k are of order the length of the
        path.

    Returns
    -------
    SigX: array, shape (n,p)
        A matrix containing in each row the signature truncated at k of a
        sample.
    """
    if k == 0:
        return np.full((np.shape(X)[0], 1), 1)
    else:
        d = X.shape[2]
        sigX = np.zeros((np.shape(X)[0], isig.siglength(d, k) + 1))
        sigX[:, 0] = 1
        for i in range(np.shape(X)[0]):
            sigX[i, 1:] = get_signature(X[i, :, :], k, norm_path=norm_path)
        return sigX
