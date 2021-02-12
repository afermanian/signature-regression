from definitions import *
import numpy as np
import pandas as pd
from simulation import DataGenerator
import skfda
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_weather():
	"""Fetch the Canadian Weather dataset from the skfda package. The input curves are the temperature curves, while the
	output Y is the log of the total precipitations.

	Returns
	-------
	X: array, shape (n,n_points,d)
	Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise linear
	paths, each composed of n_points.

	Y: array, shape (n)
		Array of target values.
	"""
	data = skfda.datasets.fetch_weather()
	fd = data['data']

	# Split dataset, temperatures and curves of precipitation
	X, Y_func = fd.coordinates
	Y = np.log10(Y_func.data_matrix.sum(axis=1)[:, 0])
	return X.data_matrix, Y


def get_electricity_loads(nclients=10):
	"""Fetch the Electricity Loads dataset, which should be stored in the DATA_DIR directory. The matrix X corresponds
	to the electricity consumption of nclients over a week, the output Y is the maximal consumption of the next week
	summed over all clients.

	Returns
	-------
	X: array, shape (n,n_points,d)
	Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise linear
	paths, each composed of n_points.

	Y: array, shape (n)
		Array of target values.
	"""
	data = pd.read_pickle(DATA_DIR + '/UCI/df_hourly_electricity_loads.pkl')

	rng = np.random.RandomState(2)
	keep_cols = rng.choice(data.shape[1], nclients)

	all_dates = pd.date_range(start='2012-01-01', end='2014-12-31', freq='7D')

	X = np.zeros((len(all_dates) - 2, data[all_dates[0]:all_dates[1]].shape[0] - 1, nclients))
	Y = np.zeros(len(all_dates) - 2)

	for i in range(len(all_dates) - 2):
		X[i, :, :] = data[all_dates[i]:all_dates[i + 1]].iloc[:-1, keep_cols].to_numpy()

		# Sum all consumption and take the maximum for the following day
		Y[i] = data[all_dates[i + 1]:all_dates[i + 2]].iloc[:-1, :].sum(axis=1).max()

	return X, Y / 10 ** 5


def get_train_test_data(X_type, ntrain=None, nval=None, Y_type=None, npoints=None, d=None, seed=None, scale_X=True):
	"""Returns the train/test splits of the various types of data used in all experiments

	Parameters
	----------
	X_type: str
		Type of functional covariates. Possible values are 'smooth_dependent', 'smooth_independent' (for the smooth
		curves with independent or dependent coordinates), 'gaussian_processes', 'weather' (for the Canadian Weather
		dataset) and 'electricity_loads' (for the Electricity Loads dataset).
	ntrain: int
		Number of training samples.
	nval: int
		Number of validation samples.
	Y_type: str
		Type of response, used only if X_type is 'smooth_dependent' or 'smooth_independent'. Possible values are
		'mean', 'max', or 'sig'.
	npoints: int
		Number of sampling points of the data.
	d: int
		Dimension of the space of the functional covariates X, from which an output Y is learned.
	seed: int
		Random seed for the generation of the smooth paths.
	scale_X: boolean
		Whether to scale the different coordinates of X to have zero mean and unit variance. This is useful if the
		orders of magnitude of the coordinates of X are very different one from another.

	Returns
	-------
	Xtrain: array, shape (ntrain, n_points, d)
		Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise linear
	paths, each composed of n_points.

	Ytrain: array, shape (ntrain)
		Array of target values for the training data.

	Xval: array, shape (nval, n_points, d)
		Array of validation paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise linear
	paths, each composed of n_points.

	Yval: array, shape (nval)
		Array of target values for the validation data.
	"""
	if X_type in ['smooth_dependent', 'smooth_independent']:
		sim = DataGenerator(npoints + 1, d, seed=seed)
		Xtrain, Ytrain = sim.get_XY_polysinus(ntrain, X_type=X_type, Y_type=Y_type)
		Xval, Yval = sim.get_XY_polysinus(nval, X_type=X_type, Y_type=Y_type)

	elif X_type == 'gp':
		sim = DataGenerator(npoints, d)
		Xtrain, Ytrain = sim.get_XY_gaussian_process(ntrain)
		Xval, Yval = sim.get_XY_gaussian_process(nval)

	elif X_type == 'weather':
		X, Y = get_weather()
		Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.33)

	elif X_type == 'electricity_loads':
		X, Y = get_electricity_loads(nclients=d)
		Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.33)

	if scale_X:
		# Scale all coordinates of X
		for i in range(Xtrain.shape[2]):
			scaler = StandardScaler()
			scaler.fit(Xtrain[:, :, i])
			Xtrain[:, :, i] = scaler.transform(Xtrain[:, :, i])
			Xval[:, :, i] = scaler.transform(Xval[:, :, i])

	return Xtrain, Ytrain, Xval, Yval
