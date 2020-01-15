from simuObjects import dataSimu, orderEstimator
import numpy as np
import matplotlib.pyplot as plt
from read_data import temp_data_to_input_matrices 
import seaborn as sns
import pandas as pd

sns.set()

# Get simulation data

d=2
nb_points=200
mast=3

max_k=11
rho=0.4

Y_noise_std=30

est=orderEstimator(d,rho=rho)

# Data for selection of alpha and Kpen
#sim=dataSimu(nb_points,5000,d)
#X=sim.get_X()
#plt.plot(np.transpose(X[:10,:,0]))
#plt.show()
#Y=sim.get_Y_sig(X,mast,Y_noise_std,plot=False)
#alpha=est.get_alpha_ridgeCV(Y,X,1,alphas=np.linspace(10**(-9),1,num=1000),plot=True)
alpha=0.01
print("alpha: ",alpha)

# Choose Kpen
#K_values=np.linspace(10**(-1),5*10**(2),num=500)
#hatm_values=est.slope_heuristic(K_values,X,Y,max_k,alpha)

# Value obtained for mast=3, 5000 data points
Kpen=80


n_grid=[10,50,100,500,1000,5000]
print(n_grid)
nb_iterations=100
hatm_values=np.zeros(len(n_grid)*nb_iterations)
n_values=np.zeros(len(n_grid)*nb_iterations)

max_k=8

for i in range(len(n_grid)):
	print("n=",n_grid[i])

	for j in range(nb_iterations):
		print("Iteration nb : ",j)
		sim=dataSimu(nb_points,n_grid[i],d)
		X=sim.get_X()
		Y=sim.get_Y_sig(X,mast,Y_noise_std)

		#if i==0 and j==0:
		#	alpha=est.get_alpha_ridgeCV(Y,X,1)
		#	print("alpha: ",alpha)
		
		hatm_values[i*nb_iterations+j]=est.get_hatm(Y,X,max_k,Kpen=Kpen,alpha=alpha)[0]
		n_values[i*nb_iterations:(i+1)*nb_iterations]=np.repeat(n_grid[i],nb_iterations)
		print('hatm is : ',hatm_values[i*nb_iterations+j])

df=pd.DataFrame({'n':n_values,'hatm':hatm_values})
print(df)

fig, ax = plt.subplots()    
# sns.stripplot(df.n, df.hatm, jitter=0.1, size=8, ax=ax, linewidth=.5)
# plt.xticks(np.arange(len(n_grid)),n_grid)
# plt.yticks(np.arange(0,max_k+1),np.arange(0,max_k+1))
# plt.xlabel("Sample size n")
# plt.ylabel(r"Estimator $\hat{m}$")
# plt.show()

sns.boxplot(x=df.n,y=df.hatm)
plt.xticks(np.arange(len(n_grid)),n_grid)
plt.yticks(np.arange(0,max_k+1),np.arange(0,max_k+1))
plt.xlabel("Sample size n")
plt.ylabel(r"Estimator $\hat{m}$")
plt.show()

