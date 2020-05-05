from read_data import canadian_weather
from simuObjects import orderEstimator
from plot_tensor_heatmap import plot_tensor_heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

#sns.set_style("white")
sns.set()

# Hyperparameters
max_k=10
rho=0.4

# Get data
X,Y=canadian_weather()

# Plot some samples
index = pd.date_range("1 1 2000", periods=365,freq="d", name="date")

fig, ax = plt.subplots()
ax.plot(index,X[:5,:,0].transpose())
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
plt.ylabel("Average daily temperatures")
plt.show()

# Find by cross validation a reasonable value for alpha, the regularization 
# parameter.
d=X.shape[2]
est=orderEstimator(d,rho=rho)
alpha=est.get_alpha_ridgeCV(Y,X,1)
print("alpha: ",alpha)

# Choose Kpen
K_values=np.linspace(10**(-8),2*10**(-1),num=200)
hatm_values=est.slope_heuristic(K_values,X,Y,max_k,alpha)
Kpen = float(input("Enter slope heuristic constant Kpen: "))
#Kpen=0.006

# Estimate hatm		
hatm=est.get_hatm(Y,X,max_k,Kpen=Kpen,alpha=alpha,plot=True)[0]
print("Hatm : ",hatm)

# Fit the linear model with hatm
reg,Ypred=est.fit_ridge(Y,X,hatm,alpha=alpha,norm_path=False)

plot_tensor_heatmap(reg.coef_[0],d,hatm)

plt.scatter(Y,Ypred)
plt.plot([2.1,3.4],[2.1,3.4],'--',color='black')
plt.xlabel("Target values")
plt.ylabel("Predicted values")
plt.show()


print("Mean absolute error : ",np.mean(np.abs(Y-Ypred)))



