import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sktime.utils.load_data import load_from_arff_to_dataframe



def canadian_weather():
	''' Loads Canadian Weather dataset and formats it into a prediction matricx 
	X with the temperature trajectories and a target value vector y with the 
	annual precipitation

	Returns
	----------
	X: array, shape (n,n_points,d)
		A 3-dimensional array, containing the coordinates in R^d of n
		piecewise linear paths, each composed of n_points. Each path is the 
		temperature profile at a location in Canada.

	y: array, shape (n)
		Array of target values, i.e. the annual precipitation in each location.
	'''
	temp_df=pd.read_csv('data/daily_temp.csv',index_col=0)
	prec_df=pd.read_csv('data/annual_prec.csv',index_col=0)
	y=np.array(prec_df)
	X=np.zeros((temp_df.shape[1],temp_df.shape[0],2))
	X[:,:,0]=np.transpose(np.array(temp_df))
	for i in range(len(y)):
		X[i,:,1]=np.linspace(0,1,num=temp_df.shape[0])
	return(X,y)


def get_ucr(name,train=True):
	''' Load a dataset from the ucr repository.

	Parameters
	----------
	name: string
		Name of the dataset to load

	train: boolean, default=True
		Whether to load the train dataset or the test dataset.

	Returns
	-------
	X: array, shape (n,n_points,d)
		A 3-dimensional array, containing the coordinates in R^d of n
		piecewise linear paths, each composed of n_points.

	y: array, shape (n)
		Array of target labels.
	'''
	if train:
		data,y=load_from_arff_to_dataframe(
			'data/ucr/{0}/{0}_TRAIN.arff'.format(name))
	else:
		data,y=load_from_arff_to_dataframe(
			'data/ucr/{0}/{0}_TEST.arff'.format(name))
	X=np.zeros((data.shape[0],len(data.iloc[0,0]),data.shape[1]+1))
	print(X.shape)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			X[i,:,j]=data.iloc[i,j]
		X[i,:,data.shape[1]]=np.linspace(0,1,num=len(data.iloc[0,0]))
	return(X,y)






