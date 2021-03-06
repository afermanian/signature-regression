from main import dataSimu, orderEstimator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os

sns.set()

# Hyperparameters
d=2
nb_points=200
mast=5
max_k=8
rho=0.4
Y_noise_std=100

est=orderEstimator(d,rho=rho)

# Get data for the selection of alpha and Kpen
sim=dataSimu(nb_points,d,mast)
X=sim.get_X(100)
Y=sim.get_Y_sig(X,Y_noise_std,plot=True)
print(Y)

# Plot some samples
#plt.plot(np.transpose(X[:10,:,0]))
#plt.show()

# Select alpha by cross validation with signatures truncated at order 1.
alpha=est.get_alpha_ridgeCV(Y,X,1,alphas=np.linspace(10**(-1),10**(4),num=1000))
print("alpha selected : ",alpha)

# Choose Kpen
K_values=np.linspace(10**(2),5*10**(5),num=500)
hatm_values=est.slope_heuristic(K_values,X,Y,max_k,alpha)
Kpen = float(input("Enter slope heuristic constant Kpen: "))

n_grid=[10,50,100,500,1000,2000,5000]
nb_iterations=100
hatm_values=np.zeros(len(n_grid)*nb_iterations)
n_values=np.zeros(len(n_grid)*nb_iterations)
pred_error=np.zeros(len(n_grid)*nb_iterations)

for i in range(len(n_grid)):
	print("n=",n_grid[i])
	n_values[i*nb_iterations:(i+1)*nb_iterations]=np.repeat(
		n_grid[i],nb_iterations)
	for j in range(nb_iterations):
		print("Iteration nb : ",j)
		X=sim.get_X(n_grid[i])
		Y=sim.get_Y_sig(X,Y_noise_std)
		
		print("get hatm")
		hatm=est.get_hatm(Y,X,max_k,Kpen=Kpen,alpha=alpha,plot=False)[0]
		
		hatm_values[i*nb_iterations+j]=hatm
		print('hatm is : ',hatm_values[i*nb_iterations+j])

		X_pred=sim.get_X(n_grid[i])
		Y_pred=sim.get_Y_sig(X_pred,Y_noise_std)

		reg=est.fit_ridge(Y,X,hatm,alpha=alpha,norm_path=False)[0]
		Y_pred_hat=est.predict_ridge(reg,X_pred,hatm)
		pred_error[i*nb_iterations+j]=np.sum((Y_pred-Y_pred_hat)**2)/len(Y_pred)


file_name="%s_cvg_hatm" % (time.strftime("%m%d-%H%M%S"))


df=pd.DataFrame(
	{'n':n_values.astype(int),r"Value of $\hat{m}$":hatm_values.astype(int),
	'pred_error':pred_error})
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


