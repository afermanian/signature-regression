import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import skfda

from definitions import *
from simulation import DataGenerator


def get_weather():
	data = skfda.datasets.fetch_weather()
	fd = data['data']

	# Split dataset, temperatures and curves of precipitation
	X, Y_func = fd.coordinates
	Y = np.log10(Y_func.data_matrix.sum(axis=1)[:, 0])
	return X.data_matrix, Y


def get_electricity_loads(nclients=10):
	data = pd.read_pickle(DATA_DIR + '/UCI/df_hourly_electricity_loads.pkl')

	rng = np.random.RandomState(2)
	keep_cols = rng.choice(data.shape[1], nclients)

	all_dates = pd.date_range(start='2012-01-01', end='2014-12-31', freq='7D')

	X = np.zeros((len(all_dates) - 2, data[all_dates[0]:all_dates[1]].shape[0] - 1, nclients))
	Y = np.zeros(len(all_dates) - 2)

	for i in range(len(all_dates) - 2):
		X[i, :, :] = data[all_dates[i]:all_dates[i+1]].iloc[:-1, keep_cols].to_numpy()

		# Sum all consumption and take the maximum for the following day
		Y[i] = data[all_dates[i+1]:all_dates[i+2]].iloc[:-1, :].sum(axis=1).max()

	return X, Y / 10 ** 5


def get_train_test_data(X_type, ntrain=None, nval=None, Y_type=None, npoints=None, d=None, noise_X_std=None,
						nclients=None, seed=None, scale_X=True):
	if X_type in ['dependent', 'independent']:
		sim = DataGenerator(npoints + 1, d, noise_std=noise_X_std, seed=seed)
		Xtrain, Ytrain = sim.get_XY_polysinus(ntrain, X_type=X_type, Y_type=Y_type)
		Xval, Yval = sim.get_XY_polysinus(nval, X_type=X_type, Y_type=Y_type)

	elif X_type in ['gp_dependent', 'gp_independent']:
		sim = DataGenerator(npoints, d, noise_std=noise_X_std)
		Xtrain, Ytrain = sim.get_XY_gaussian_process(ntrain, X_type=X_type)
		Xval, Yval = sim.get_XY_gaussian_process(nval, X_type=X_type)

	elif X_type == 'weather':
		X, Y = get_weather()
		Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.33)

	elif X_type == 'electricity_loads':
		X, Y = get_electricity_loads(nclients=nclients)
		Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.33)

	if scale_X:
		# Scale all coordinates of X
		for i in range(Xtrain.shape[2]):
			scaler = StandardScaler()
			scaler.fit(Xtrain[:, :, i])
			Xtrain[:, :, i] = scaler.transform(Xtrain[:, :, i])
			Xval[:, :, i] = scaler.transform(Xval[:, :, i])

	return Xtrain, Ytrain, Xval, Yval
