from read_data import canadian_weather
from simuObjects import orderEstimator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

def plot_tensor_heatmap(x,d,k):
	#print(x.shape)
	mat_coef=np.zeros((k+1,d**k))
	mask=np.zeros((k+1,d**k))
	count=0
	for j in range(k+1):
		mat_coef[j,:d**j]=x[count:count+d**j]
		mask[j,d**j:]=True
		count+=d**j
	#print(mat_coef)
	#print(mask)
	with sns.axes_style("white"):
		f, ax = plt.subplots()
		ax = sns.heatmap(mat_coef, mask=mask, vmax=.3,xticklabels=False,
			center=0,cbar_kws={"orientation": "horizontal"})
	plt.show()





