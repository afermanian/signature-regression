import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def temp_data_to_input_matrices(p=10):
	df=pd.read_csv('data/daily-minimum-temperatures-in-me.csv')
	ts=np.array(df.ix[:,1].values,dtype=float)
	nb_points=len(ts)
	X=np.zeros((nb_points-p,p,2))
	Y=np.zeros(nb_points-p)
	for i in range(nb_points-p):
		X[i,:,0]=ts[i:i+p]
		Y[i]=ts[i+p]
		X[i,:,1]=np.linspace(0,1,num=p)
	return(X,Y)


def canadian_weather():
	temp_df=pd.read_csv('data/daily_temp.csv',index_col=0)
	prec_df=pd.read_csv('data/annual_prec.csv',index_col=0)
	y=np.array(prec_df)
	X=np.zeros((temp_df.shape[1],temp_df.shape[0],2))
	X[:,:,0]=np.transpose(np.array(temp_df))
	for i in range(len(y)):
		X[i,:,1]=np.linspace(0,1,num=temp_df.shape[0])
	return(X,y)


def metro_traffic(p=100):
	df=pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv',parse_dates=['date_time'],index_col=['date_time'])
	df=df['2015-06-11':]
	print(df.head())

	# Clean missing values/outliers	
	df.loc[df.rain_1h>100,'rain_1h']=df.rain_1h.mean()

	# Create input matrices
	nb_points=df.shape[0]
	Y=np.zeros(nb_points-p)
	X=np.zeros((nb_points-p,p,5))
	for i in range(nb_points-p):
		X[i,:,0]=df.temp[i:i+p]
		X[i,:,1]=df.rain_1h[i:i+p]
		X[i,:,2]=df.snow_1h[i:i+p]
		X[i,:,3]=df.traffic_volume[i:i+p]
		X[i,:,4]=np.linspace(0,1,num=p)
		Y[i]=df.traffic_volume[i+p]
	return(X,Y)








