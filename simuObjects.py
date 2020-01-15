import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import iisignature as isig
from sklearn.linear_model import RidgeCV,Ridge
from sklearn.linear_model import LassoCV,Lasso

sns.set()

def get_curve_length(path):
	length=0
	for i in range(path.shape[0]-1):
		length+=math.sqrt(np.sum((path[i+1,:]-path[i,:])**2))
	return(length)


def get_signature(path,order):
	#path=path/get_curve_length(path)
	return(isig.sig(path,order))


class orderEstimator(object):
	def __init__(self,d,rho=1/4):
		self.rho=rho
		self.d=d

	def get_penalization(self,n,k,Kpen=1):
		return(Kpen*n**(-self.rho)*math.sqrt(isig.siglength(self.d,k)))

	def get_SigX(self,X,k):
		SigX=np.zeros((np.shape(X)[0],isig.siglength(self.d,k)))
		print("Create signature matrix")
		for i in range(np.shape(X)[0]):
			SigX[i,:]=get_signature(X[i,:,:],k)
		print("Input matrix size: ",SigX.shape)
		max_SigX=np.amax(np.absolute(SigX),axis=0)
		SigX=SigX/max_SigX
		return(SigX)


	def get_alpha_ridgeCV(self,Y,X,k,plot=False,alphas=np.linspace(10**(-6),10,num=1000)):
		#reg=LassoCV()
		reg=RidgeCV(alphas=alphas,store_cv_values=True)
		SigX=self.get_SigX(X,k)
		reg.fit(SigX,Y)
		print("alpha ridge cv: ",reg.alpha_)
		if plot:
			plt.plot(alphas,np.mean(reg.cv_values_,axis=0))
			plt.show()
		return(reg.alpha_)


	def get_hatL(self,Y,X,k,alpha=1,plot=False):
		reg=Ridge(alpha=alpha,normalize=False)
		SigX=self.get_SigX(X,k)
		print("Fit ridge regression")
		reg.fit(SigX,Y)
		print("End fitting")
		Ypred=reg.predict(SigX)
		if plot:
			plt.plot(reg.coef_)
			plt.title("Regression coefficients")
			plt.show()
			plt.scatter(Y,Ypred)
			plt.title("Ypred against Y")
			plt.show()
		return(np.sum((Y-Ypred)**2)/len(Y))

	def get_hatm(self,Y,X,max_k,Kpen=1,alpha=1,plot=False):
		objective=np.zeros(max_k)
		loss=np.zeros(max_k)
		pen=np.zeros(max_k)
		for i in range(max_k):
			loss[i]=self.get_hatL(Y,X,i+1,alpha=alpha)
			pen[i]=self.get_penalization(Y.shape[0],i+1,Kpen=Kpen)
			objective[i]=loss[i]+pen[i]
		hatm=np.argmin(objective)+1

		if plot:
			plt.plot(np.arange(max_k)+1,loss,label="loss")
			plt.plot(np.arange(max_k)+1,pen,label="penalization")
			plt.plot(np.arange(max_k)+1,objective,label="sum")
			plt.legend()
			plt.show()
		return(hatm,objective)

	def slope_heuristic(self,K_values,X,Y,max_k,alpha):
		hatm=np.zeros(len(K_values))
		loss=np.zeros(max_k)
		for j in range(max_k):
			loss[j]=self.get_hatL(Y,X,j+1,alpha=alpha)

		for i in range(len(K_values)):
			#print(i)
			pen=np.zeros(max_k)
			for j in range(max_k):
				pen[j]=self.get_penalization(Y.shape[0],j+1,Kpen=K_values[i])
			hatm[i]=np.argmin(loss+pen)+1
			#print("Hatm selected: ",hatm[i])

		print(hatm)
		print(K_values)

		# Plot
		fig, ax = plt.subplots()
		jump=1
		for i in range(1,max_k+1):
			if i in hatm:
				xmin=K_values[hatm==i][0]
				xmax=K_values[hatm==i][-1]
				ax.hlines(i,xmin,xmax,colors='b')
				if i!=1:
					ax.vlines(xmax,i,i-jump,linestyles='dashed',colors='b')
				jump=1
			else:
				jump+=1
		ax.set(xlabel=r'$K_{pen}$',ylabel=r'$\hat{m}$')
		plt.show()

		return(hatm)

		
class dataSimu(object):
	def __init__(self,nb_points,nb_simu,d):
		self.nb_points=nb_points
		self.nb_simu=nb_simu
		self.d=d

	def get_X(self):
		X=np.zeros((self.nb_simu,self.nb_points,self.d))
		Xlength=np.zeros((self.nb_simu,self.nb_points,self.d))
		times=np.linspace(0,1,num=self.nb_points)
		for i in range(self.nb_simu):
			for j in range(self.d-1):
				param=np.random.random(size=4)
				X[i,:,j]=param[0]+ 10*param[1]*np.sin(times*np.pi*2/param[2]) + 10*(times-param[3])**3
			X[i,:,self.d-1]=times
			Xlength[i,:,:]=np.full((self.nb_points,self.d),get_curve_length(X[i,:,:]))
		#plt.plot(X[0,:,:])
		#plt.show()
		return(X/Xlength)

	def get_Y_sig(self,X,order,noise_std,plot=False):
		Y=np.zeros(self.nb_simu)
		noise=np.random.normal(scale=noise_std,size=self.nb_simu)
		size_sig=isig.siglength(self.d,order)

		SigX=np.zeros((self.nb_simu,size_sig))

		for i in range(self.nb_simu):
			SigX[i,:]=get_signature(X[i,:,:],order)

		#beta=np.exp(np.linspace(0,10,num=size_sig))
		#beta=np.full(size_sig,1)
		beta=np.random.random(size=size_sig)
		beta=beta/math.sqrt(np.sum(beta**2))

		beta_repeated=np.repeat(beta.reshape(1,size_sig),self.nb_simu,axis=0)

		#Y=np.sum(X[:,:,0],axis=1)
		#Y=Y/math.sqrt(np.sum(Y**2))
		#print(np.shape(Y))
		Y=1000*np.sum(beta_repeated*SigX,axis=1)
		if plot:
			plt.plot(SigX[0,:])
			plt.plot(beta)
			plt.title("SigX and beta")
			plt.show()
			plt.scatter(Y,Y+noise)
			plt.title("Y against Y+noise")
			plt.show()
		return(Y+noise)

	def get_Y_nonlinear(self,X,noise_std):
		Y=np.zeros(self.nb_simu)
		for i in range(self.nb_simu):
			Y[i]+=np.mean(np.prod(X[i,:,:],axis=1))*10000
		noise=np.random.normal(scale=noise_std,size=self.nb_simu)
		#plt.scatter(Y,Y+noise)
		#plt.show()
		return(Y+noise)

















