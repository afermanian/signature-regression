from simuObjects import dataSimu, orderEstimator
import numpy as np
import matplotlib.pyplot as plt
from read_data import temp_data_to_input_matrices 
import seaborn as sns
import pandas as pd
import time
import os

sns.set()

# Get simulation data

d=2
nb_points=200
mast=5

max_k=7
rho=0.4

Y_noise_std=10

est=orderEstimator(d,rho=rho)

# Data for selection of alpha and Kpen
sim=dataSimu(nb_points,d,mast)
X=sim.get_X(1000)
#plt.plot(np.transpose(X[:10,:,0]))
#plt.show()
Y=sim.get_Y_sig(X,Y_noise_std,plot=True)
alpha=est.get_alpha_ridgeCV(Y,X,1,alphas=np.linspace(10**(-9),10**(-1),num=1000))


# Choose Kpen
#K_values=np.linspace(10**(-5),5*10**(2),num=500)
#hatm_values=est.slope_heuristic(K_values,X,Y,max_k,alpha)

Kpen=1
#hatm,loss=est.get_hatm(Y,X,max_k,Kpen=Kpen,alpha=alpha)
#print("hatm : ",hatm)


# Value obtained for mast=3, 5000 data points : Kpen=80


n_grid=[10,50,100,500,1000,5000,10000]
print(n_grid)
nb_iterations=100
hatm_values=np.zeros(len(n_grid)*nb_iterations)
n_values=np.zeros(len(n_grid)*nb_iterations)
pred_error=np.zeros(len(n_grid)*nb_iterations)

for i in range(len(n_grid)):
	print("n=",n_grid[i])
	n_values[i*nb_iterations:(i+1)*nb_iterations]=np.repeat(n_grid[i],nb_iterations)
	for j in range(nb_iterations):
		print("Iteration nb : ",j)
		X=sim.get_X(n_grid[i])
		Y=sim.get_Y_sig(X,Y_noise_std)

		#if i==0 and j==0:
		#	alpha=est.get_alpha_ridgeCV(Y,X,1)
		#	print("alpha: ",alpha)

		hatm=est.get_hatm(Y,X,max_k,Kpen=Kpen,alpha=alpha)[0]
		
		hatm_values[i*nb_iterations+j]=hatm
		print('hatm is : ',hatm_values[i*nb_iterations+j])

		X_pred=sim.get_X(n_grid[i])
		Y_pred=sim.get_Y_sig(X,Y_noise_std)

		pred_error[i*nb_iterations+j]=est.get_hatL(Y,X,hatm,alpha=alpha)/n_grid[i]


file_name="%s_cvg_hatm" % (time.strftime("%m%d-%H%M%S"))


df=pd.DataFrame({'n':n_values.astype(int),r"Value of $\hat{m}$":hatm_values.astype(int),'pred_error':pred_error})
print(df)
df.to_csv(os.path.join('results',file_name+'.csv'))


file = open(os.path.join('results',file_name+'.txt'),"w")
file.write("d: %s \n" % (d))
file.write('nb_points: %s \n' % (nb_points))
file.write('mast: %s \n' %(mast))
file.write('max_k:%s \n' % (max_k))
file.write('rho: %s \n' % (rho))
file.write('Y_noise_std: %s \n' %(Y_noise_std))
file.write('alpha: %s \n' %(alpha))
file.write('Kpen: %s \n' % (Kpen))
file.write('nb_iterations: %s \n' % (nb_iterations))
file.close()

g=sns.catplot(x='n', hue=r"Value of $\hat{m}$", data=df,kind="count")
plt.xlabel(r"Sample size $n$")
plt.savefig(os.path.join('results',file_name+'_hist_hatm'+'.png'))
plt.show()

# fig, ax = plt.subplots()    
# sns.stripplot(df.n, df.hatm, jitter=0.1, size=8, ax=ax, linewidth=.5)
# plt.xticks(np.arange(len(n_grid)),n_grid)
# plt.yticks(np.arange(0,max_k+1),np.arange(0,max_k+1))
# plt.xlabel("Sample size n")
# plt.ylabel(r"Estimator $\hat{m}$")
# plt.show()

sns.boxplot(x=df.n,y=df[r"Value of $\hat{m}$"])
plt.xticks(np.arange(len(n_grid)),n_grid)
plt.yticks(np.arange(0,max_k+1),np.arange(0,max_k+1))
plt.xlabel("Sample size n")
plt.ylabel(r"Estimator $\hat{m}$")
plt.savefig(os.path.join('results',file_name+'_boxplot_hatm'+'.png'))
plt.show()


sns.boxplot(x=df.n,y=df.pred_error)
plt.xticks(np.arange(len(n_grid)),n_grid)
plt.xlabel("Sample size n")
plt.ylabel(r"Prediction error")
plt.savefig(os.path.join('results',file_name+'_boxplot_pred_errror'+'.png'))
plt.show()