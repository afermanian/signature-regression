import numpy as np
import iisignature as isig
import matplotlib.pyplot as plt
import seaborn as sns
from skfda.misc.covariances import Exponential
from skfda.datasets import make_gaussian_process

from tools import get_sigX

sns.set()


# noinspection PyPep8Naming
class DataGenerator(object):
	"""Generate some training data.

	Parameters
	----------
	n_points: int
		Number of points in the piecewise linear approximations.

	d: int
		Dimension of the output space of the training paths.

	noise_std: float or None
		Variance of the Gaussian noise on X
	"""

	def __init__(self, npoints, d, noise_std=0, seed=None):
		self.npoints = npoints
		self.d = d
		self.noise_std = noise_std
		if seed:
			np.random.seed(seed)

	def get_X_polysinus(self, n, X_type):
		""" Generates n sample paths X:[0,1] -> R^d, defined by
		X_t=alpha_1 + 10*alpha_2*sinus(2*pi*t/alpha_3) + 10*(t-alpha_4)^3,
		where the alphas are sampled uniformly over [0,1].X is interpolated by
		a piecewise linear function with n_points. Each sample X is normalized
		so that it has length 1.

		Parameters
		----------
		n: int
			Number of samples to simulate.

		Returns
		-------
		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.
		"""
		X = np.zeros((n, self.npoints, self.d))

		times = np.linspace(0, 1, num=self.npoints)
		for i in range(n):
			if X_type == 'dependent':
				sinus_param = np.random.random(size=4)
			for j in range(self.d):
				if X_type == 'dependent':
					param = np.random.random() * sinus_param
				else:
					param = np.random.random(size=4)
				X[i, :, j] = param[0] + 10 * param[1] * np.sin(
					times * np.pi * 2 / param[2]) + 10 * (times - param[3]) ** 3
		return X

	def get_XY_gaussian_process(self, n, X_type='gp_independent'):
		X = np.zeros((n, self.npoints, self.d))
		times = np.repeat(np.expand_dims(np.linspace(0, 1, self.npoints), -1), n * self.d, 1)
		times = times.reshape((self.npoints, n, self.d)).transpose((1, 0, 2))

		if X_type == 'gp_independent':
			slope = 3 * (2 * np.random.random((n, self.d)) - 1)

		elif X_type == 'gp_dependent':
			alpha = 3 * (2 * np.random.random(n) - 1)
			alpha = np.repeat(np.expand_dims(alpha, -1), self.d, 1)
			slope = alpha * np.random.random((n, self.d))

		else:
			raise NameError('X_type not well specified')

		Y = np.sqrt(np.sum(slope ** 2, axis=1))
		slope = np.repeat(np.expand_dims(slope, 0), self.npoints, 0).transpose((1, 0, 2))
		for i in range(n):
			gp = make_gaussian_process(n_features=self.npoints, n_samples=self.d, cov=Exponential())
			X[i, :, :] = gp.data_matrix.T[0]

		X = X + slope * times

		return X, Y

	def get_Y_sig(self, X, mast, noise_std=100, plot=False):
		"""Compute the target values Y as scalar products of the truncated
		signatures of rows of X with a certain parameter beta plus a gaussian
		noise. Y follows therefore the expected signature model.

		Parameters
		----------
		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		mast: int
			True value of the truncation order of the signature.

		noise_std: float
			Amount of noise of Y, Y is equal to the scalar product of the
			signature of X against beta plus a uniform noise on
			[-noise_std,noise_std].

		plot: boolean, default=False
			If True, output two plots: one plot with the signature coefficients
			ofone sample and the regression vector beta, one scatter plot with Y
			against Y+noise to check the calibration of the noise variance.

		Returns
		-------
		Y: array, shape (n)
			Target values, defined by the expected signature model.
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

	def get_XY_polysinus(self, ntrain, X_type='independent', Y_type='mean', mast=5):

		Xraw = self.get_X_polysinus(ntrain, X_type)

		if Y_type == 'mean':
			Y = np.mean(Xraw[:, -1, :], axis=1)
		elif Y_type == 'max':
			Y = np.max(Xraw[:, -1, :], axis=1)
		elif Y_type == 'sig':
			Y = self.get_Y_sig(Xraw[:, :-1, :], mast, noise_std=10)
		else:
			raise NameError('Y_type not well specified')

		X = Xraw[:, :-1, :]
		noise = 2 * self.noise_std * np.random.random(size=X.shape) - self.noise_std
		return X + noise, Y