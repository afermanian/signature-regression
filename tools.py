import iisignature as isig
import numpy as np


def add_time(X):
	times = np.tile(np.linspace(0, 1, X.shape[1]), (X.shape[0], 1))
	Xtime = np.concatenate([X, times.reshape((times.shape[0], times.shape[1], 1))], axis=2)
	return Xtime


def get_sigX(X, k):
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
			sigX[i, 1:] = isig.sig(X[i, :, :], k)
		return sigX

