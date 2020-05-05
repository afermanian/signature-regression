from read_data import get_ucr
from simuObjects import orderEstimator
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plot_tensor_heatmap import plot_tensor_heatmap

max_k=11
rho=0.4
plot=False

sns.set()
name='BirdChicken'

# Load dataset
X,Y_raw=get_ucr(name)

le=LabelEncoder()
Y=le.fit_transform(Y_raw)

# Set plot=True if you want to plot some samples
if plot:
	fig, ax = plt.subplots()
	ax.plot(X[:5,:,0].transpose())
	plt.show()


d=X.shape[2]
est=orderEstimator(d,rho=rho)
alpha=est.get_alpha_ridgeCV(Y,X,1,plot=False,alphas=np.logspace(-3,9,num=1000))
#alpha=10**(-6)
print("alpha: ",alpha)

# Choose Kpen
K_values=np.logspace(-15,0,num=1000)
hatm_values=est.slope_heuristic(K_values,X,Y,max_k,alpha)
Kpen = float(input("Enter slope heuristic constant Kpen: "))

#Estimate hatm
hatm=est.get_hatm(Y,X,max_k,Kpen=Kpen,alpha=alpha,plot=True)[0]
print("Hatm : ",hatm)

reg,Ypred=est.fit_ridge(Y,X,hatm,alpha=alpha)

X_test,Y_raw_test=get_ucr(name,train=False)
Y_test=le.transform(Y_raw_test)

Y_test_pred=est.predict_ridge(reg,X_test,hatm)


Ypred_cat=(Ypred>0.5).astype(int)
Y_test_pred_cat=(Y_test_pred>0.5).astype(int)

print("Accuracy on training set : ",np.mean(Ypred_cat==Y))
print("Accuracy on test set : ",np.mean(Y_test_pred_cat==Y_test))

plot_tensor_heatmap(reg.coef_,d,hatm)






