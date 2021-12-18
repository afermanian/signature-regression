import iisignature as isig
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skfda.misc.covariances import Exponential
from skfda.datasets import make_gaussian_process
from tools import get_sigX

sns.set()


class DataGenerator(object):
	"""Generate some training data.

	Parameters
	----------
	npoints: int
		Number of sampling points of the data.

	d: int
		Dimension of the output space of the training paths.

	noise_std: float or None
		Variance of the Gaussian noise on X.

	seed:int
		Random seed.
	"""

	def __init__(self, npoints, d, noise_std=0, seed=None):
		self.npoints = npoints
		self.d = d
		self.noise_std = noise_std
		if seed:
			np.random.seed(seed)

	def get_X_polysinus(self, n):
		""" Generates n sample paths X:[0,1] -> R^d, defined by
		X_t=alpha_1 + 10*alpha_2*sinus(2*pi*t/alpha_3) + 10*(t-alpha_4)^3. The alphas are sampled uniformly over [0,1].

		Parameters
		----------
		n: int
			Number of samples.

		Returns
		-------
		X: array, shape (n, self.npoints, self.d)
			Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
			linear paths, each composed of npoints.
		"""
		X = np.zeros((n, self.npoints, self.d))

		times = np.linspace(0, 1, num=self.npoints)
		for i in range(n):
			for j in range(self.d):
				param = np.random.random(size=4)
				X[i, :, j] = param[0] + 10 * param[1] * np.sin(
					times * np.pi * 2 / param[2]) + 10 * (times - param[3]) ** 3
		return X

	def get_XY_gaussian_process(self, n):
		""" Generates n gaussian processes with a linear trend

		Parameters
		----------
		n: int
			Number of samples.

		Returns
		-------
		X: array, shape (n, self.npoints, self.d)
			Array of sample paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
			linear paths, each composed of npoints.
		"""
		X = np.zeros((n, self.npoints, self.d))
		times = np.repeat(np.expand_dims(np.linspace(0, 1, self.npoints), -1), n * self.d, 1)
		times = times.reshape((self.npoints, n, self.d)).transpose((1, 0, 2))
		slope = 3 * (2 * np.random.random((n, self.d)) - 1)

		Y_no_noise = np.sqrt(np.sum(slope ** 2, axis=1))
		slope = np.repeat(np.expand_dims(slope, 0), self.npoints, 0).transpose((1, 0, 2))
		for i in range(n):
			gp = make_gaussian_process(n_features=self.npoints, n_samples=self.d, cov=Exponential())
			X[i, :, :] = gp.data_matrix.T[0]

		X = X + slope * times
		noise = 2 * self.noise_std * np.random.random(size=Y_no_noise.shape) - self.noise_std
		Y = Y_no_noise + noise
		return X, Y

	def get_Y_sig(self, X, mast, noise_std=100, plot=False):
		"""Compute the target values Y as scalar products of the truncated signatures of rows of X with a certain
		parameter beta plus a gaussian noise. Y follows therefore the signature linear model.

		Parameters
		----------
		X: array, shape (n, self.npoints, self.d)
			Array of paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise linear paths,
			each composed of npoints.

		mast: int
			True value of the truncation order of the signature.

		noise_std: float, default=100
			Amount of noise of Y, Y is equal to the scalar product of the signature of X against beta plus a uniform
			noise on [-noise_std,noise_std].

		plot: boolean, default=False
			If True, output two plots: one plot with the signature coefficients of one sample and the regression vector
			beta, one scatter plot with Y against Y+noise to check the calibration of the noise variance.

		Returns
		-------
		Y: array, shape (n)
			Target values
		"""
		n = X.shape[0]

		if mast == 0:
			size_sig = 1
		else:
			size_sig = isig.siglength(self.d, mast) + 1
		beta = np.random.random(size=size_sig) / 1000
		noise = 2 * noise_std * np.random.random(size=n) - noise_std

		SigX = get_sigX(X, mast)
		beta_repeated = np.repeat(beta.reshape(1, size_sig), n, axis=0)
		Y = np.sum(beta_repeated * SigX, axis=1)

		if plot:
			plt.scatter(Y, Y + noise)
			plt.title("Y against Y+noise")
			plt.show()
		return Y + noise

	def get_XY_polysinus(self, n, Y_type='mean', mast=5):
		""" Generates n samples (X, Y) where X are smooth curves with independent or dependent coordinate, and Y is
	either the mean or the max at the next time step.

		Parameters
		----------
		n: int
			Number of samples.

		X_type: str, default='smooth_independent'
			Type of functional covariates. Possible values are 'smooth_dependent', 'smooth_independent'.

		Y_type: str, default='mean'
			Type of response. Possible values are 'mean' or 'sig'.

		mast: int
			True value of the truncation order of the signature. Used only if Y_type='sig'

		Returns
		-------
		X: array, shape (n, self.npoints, self.d)
			Array of sample paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
			linear paths, each composed of npoints.

		Y: array, shape (n)
			Target values
		"""
		Xraw = self.get_X_polysinus(n)

		if Y_type == 'mean':
			noise = 2 * self.noise_std * np.random.random(size=Xraw.shape[0]) - self.noise_std
			Y_no_noise = np.mean(Xraw[:, -1, :], axis=1)
			Y = Y_no_noise + noise
		elif Y_type == 'sig':
			Y = self.get_Y_sig(Xraw[:, :-1, :], mast, noise_std=10)
		else:
			raise NameError('Y_type not well specified')

		X = Xraw[:, :-1, :]
		return X, Y
