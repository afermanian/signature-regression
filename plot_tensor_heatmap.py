from read_data import canadian_weather
from simuObjects import orderEstimator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_tensor_heatmap(x,d,k):
	print(x)
	print((d**(k+1)-1)/(d-1))
	mat_coef=np.zeros((k+1,d**k))
	mask=np.zeros((k+1,d**k))
	count=0
	for j in range(k+1):
		mat_coef[j,:d**j]=x[count:count+d**j]
		mask[j,d**j:]=True
		count+=d**j
	print(mat_coef)
	print(mask)
	with sns.axes_style("white"):
		f, ax = plt.subplots()
		ax = sns.heatmap(mat_coef, mask=mask, vmax=.3,xticklabels=False)
	plt.show()


sns.set()

# Get data
X,Y=canadian_weather()

d=X.shape[2]
est=orderEstimator(d)
#alpha=est.get_alpha_ridgeCV(Y,X,1)
#print("alpha: ",alpha)
alpha=0.00001

hatm=5

reg,Ypred=est.fit_ridge(Y,X,hatm,alpha=alpha)
coeff=np.append([reg.intercept_],reg.coef_)

plot_tensor_heatmap(coeff,d,hatm)