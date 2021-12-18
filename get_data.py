from definitions import *
import numpy as np
import pandas as pd
from simulation import DataGenerator
from sklearn.model_selection import train_test_split
import os


def get_air_quality(univariate_air_quality=False):
	data = pd.read_csv(os.path.join(DATA_DIR, 'UCI', 'AirQualityUCI', 'AirQualityUCI.csv'), sep=';', header=0)

	# Data cleaning
	data = data.dropna(how='all')
	data[['T', 'RH', 'AH', 'CO(GT)']] = data[['T', 'RH', 'AH', 'CO(GT)']].apply(lambda x: x.str.replace(',', '.'))
	data['Hour'] = data['Time'].str.split('.', expand=True)[0]
	data['DateHour'] = data['Date'] + '/' + data['Hour']
	date_time_index = pd.to_datetime(data['DateHour'], format='%d/%m/%Y/%H')
	data.index = date_time_index

	data = data[
		['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'NO2(GT)']]
	data = data.astype(float)
	data = data.replace(to_replace=-200, value=None)
	data = data.fillna(method='ffill')

	if univariate_air_quality:
		keep_cols = ['PT08.S4(NO2)']
	else:
		keep_cols = ['PT08.S4(NO2)', 'T', 'RH']

	list_X = []
	list_Y = []

	window_length = 24 * 7 + 1
	for window in data.rolling(window=window_length):
		if len(window) == window_length:
			list_X.append(window[keep_cols][:-1].to_numpy())
			list_Y.append(window['NO2(GT)'][-1])

	X = np.stack(list_X)
	Y = np.stack(list_Y)
	return X, Y / 100


def get_train_test_data(X_type, ntrain=None, nval=None, Y_type=None, npoints=None, d=None,
						univariate_air_quality=False):
	"""Returns the train/test splits of the various types of data used in all experiments

	Parameters
	----------
	X_type: str
		Type of functional covariates. Possible values are 'smooth', 'gp', and 'air_quality'.
	ntrain: int
		Number of training samples.
	nval: int
		Number of validation samples.
	Y_type: str
		Type of response, used only if X_type is 'smooth'. Possible values are 'mean' and 'sig'.
	npoints: int
		Number of sampling points of the data.
	d: int
		Dimension of the space of the functional covariates X, from which an output Y is learned.
	seed: int
		Random seed for the generation of the smooth paths.

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
	if X_type == 'smooth':
		sim = DataGenerator(npoints + 1, d, noise_std=1.)
		Xtrain, Ytrain = sim.get_XY_polysinus(ntrain, Y_type=Y_type)
		Xval, Yval = sim.get_XY_polysinus(nval, Y_type=Y_type)

	elif X_type == 'gp':
		sim = DataGenerator(npoints, d, noise_std=1.)
		Xtrain, Ytrain = sim.get_XY_gaussian_process(ntrain)
		Xval, Yval = sim.get_XY_gaussian_process(nval)

	elif X_type == 'air_quality':
		X, Y = get_air_quality(univariate_air_quality=univariate_air_quality)
		Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.33)

	else:
		raise NameError('X_type not well specified')

	return Xtrain, Ytrain, Xval, Yval

