import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import iisignature as isig
from sklearn.linear_model import RidgeCV,Ridge

sns.set()


def get_signature(path,order,norm_path=True):
	''' Returns the signature of a path truncated at a certain order.

	Parameters
	----------
	path: array, shape (n_points,d)
		The array storing the n_points coordinates in R^d that constitute a 
		piecewise linear path.

	order: int
		The truncation order of the signature

	norm_path: boolean, default=True
		Whether to normalise the path before computing the signature, such that
		the signature coefficients of order k are of order the length of the 
		path.

	Returns
	-------
	sig: array, shape (p)
		Array containing the truncated signature coefficients of path. It is of
		shape p=(d^(order+1)-1)/(d-1)
	'''
	if norm_path:
		path=path*(math.factorial(order))**(1/order)
	return(isig.sig(path,order))


def get_SigX(X,k,norm_path=True):
	'''Returns a matrix containing signatures truncated at k of n samples
	given in the input tensor X.

	Parameters
	----------
	X: array, shape (n,n_points,d)
		A 3-dimensional array, containing the coordinates in R^d of n
		piecewise linear paths, each composed of n_points.

	k: int
		Truncation order of the signature

	norm_path: boolean, default=True
		Whether to normalise the path before computing the signature, such that
		the signature coefficients of order k are of order the length of the 
		path.

	Returns
	-------
	SigX: array, shape (n,p)
		A matrix containing in each row the signature truncated at k of a
		sample. 
	'''
	if k==0:
		return(np.full((np.shape(X)[0],1),1))
	else:
		d=X.shape[2]
		SigX=np.zeros((np.shape(X)[0],isig.siglength(d,k)+1))
		SigX[:,0]=1
		for i in range(np.shape(X)[0]):
			SigX[i,1:]=get_signature(X[i,:,:],k,norm_path=norm_path)
		return(SigX)

class orderEstimator(object):
	''' Object that implements the estimation of the truncation order.

	Parameters
	----------
	d: int
		Dimension of the space of the paths X, from which an output Y is
		learned.
	
	rho: float
		Parameter of the penalization: power of 1/n. It should satisfy :
		0<rho<1/2.
	'''
	def __init__(self,d,rho=1/4):
		self.rho=rho
		self.d=d

	def get_penalization(self,n,k,Kpen=1):
		'''Returns the penalization function used in the estimator of the
		truncation order, that is, 
		pen_n(k)=Kpen*sqrt((d^(k+1)-1)/(d-1))/n^rho.

		Parameters
		----------
		n: int
			Number of samples.

		k: int
			Truncation order of the signature.

		Kpen: float, default=1
			Constant in front of the penalization, it has to be a positive
			number.

		Returns
		-------
		pen_n(k):float
			The penalization pen_n(k)
		'''
		if k==0:
			size_sig=1
		else:
			size_sig=isig.siglength(self.d,k)+1

		return(Kpen*n**(-self.rho)*math.sqrt(size_sig))


	def get_alpha_ridgeCV(
			self,Y,X,k,plot=False,alphas=np.linspace(10**(-6),10,num=1000)):
		'''Gets the best regularization parameter for a Ridge regression on
		signatures truncated at k, by scikit-learn built-in cross-validation.

		Parameters
		----------
		Y: array, shape (n)
			Array of target values.

		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		k: int
			Truncation order of the signature

		plot: boolean, default=False
			If True, plots the errors against the regularization parameters.

		alphas: array, default=np.linspace(10**(-6),10,num=1000)
			Values of the regularization paramete to test.

		Returns
		-------
		alpha: float
			The best regularization parameter among alphas.
		'''
		
		reg=RidgeCV(alphas=alphas,store_cv_values=True,fit_intercept=False,
			gcv_mode='svd')
		SigX=get_SigX(X,k)

		reg.fit(SigX,Y)
		if plot:
			print(alphas.shape,reg.cv_values_.shape)
			plt.plot(alphas,np.mean(reg.cv_values_,axis=0)[0,:])
			plt.show()
		return(reg.alpha_)

	def fit_ridge(self,Y,X,k,alpha=1,norm_path=True):
		'''Fit a signature ridge regression.

		Parameters
		----------
		Y: array, shape (n)
			Array of target values.

		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		k: int
			Truncation order of the signature

		alpha: float, default=1
			Regularization parameter in the Ridge regression.

		norm_path:boolean, default=True
			Whether to normalize the path by (k!)^(1/k) before computing 
			signatures

		Returns
		-------
		reg: object
			Instance of sklearn.linear_model.Ridge

		Ypred: array, shape (n)
			Array of predicted values.
		'''

		reg=Ridge(alpha=alpha,normalize=False,fit_intercept=False,solver='svd')
		SigX=get_SigX(X,k,norm_path=norm_path)
		reg.fit(SigX,Y)
		Ypred=reg.predict(SigX)
		return(reg,Ypred)

	def get_hatL(self,Y,X,k,alpha=1,norm_path=True,plot=False):
		'''Computes the minimum empirical squared loss obtained with a Ridge
		regression on signatures truncated at k.

		Parameters
		----------
		Y: array, shape (n)
			Array of target values.

		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		k: int
			Truncation order of the signature

		alpha: float, default=1
			Regularization parameter in the Ridge regression.

		norm_path:boolean, default=True
			Whether to normalize the path by (k!)^(1/k) before computing 
			signatures

		plot: boolean, default=False
			If True, plots the regression coefficients and a scatter plot of the
			target values Y against its predicted values Ypred to assess the
			quality
			of the fit.

		Returns
		-------
		hatL: float
			The squared loss, that is the sum of the squares of Y-Ypred, where
			Ypred are the fitted values of the Ridge regression of Y against
			signatures of X truncated at k.
		'''
		
		reg,Ypred=self.fit_ridge(Y,X,k,alpha=alpha,norm_path=norm_path)
		if plot:
			plt.plot(reg.coef_)
			plt.title("Regression coefficients")
			plt.show()
			plt.scatter(Y,Ypred)
			plt.title("Ypred against Y")
			plt.show()
		return(np.sum((Y-Ypred)**2)/len(Y))


	def predict_ridge(self,reg,X,k,norm_path=True):
		'''Outputs prediction of a reg object, already trained with signatures 
		truncated at order k.

		Parameters
		----------
		reg: object
			Instance of sklearn.linear_model.Ridge

		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		k: int
			Truncation order of the signature

		norm_path:boolean, default=True
			Whether to normalize the path by (k!)^(1/k) before computing 
			signatures

		Returns
		-------
		Ypred: array, shape (n)
			Array of predicted values.
		'''
		SigX=get_SigX(X,k,norm_path=norm_path)
		Ypred=reg.predict(SigX)
		return(Ypred)


	def get_hatm(self,Y,X,max_k,Kpen=1,alpha=1,plot=False):
		'''Computes the estimator of the truncation order by minimizing the sum
		of hatL and the penalization, over values of k from 1 to max_k.

		Parameters
		----------
		Y: array, shape (n)
			Array of target values.

		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		max_k: int,
			Maximal value of truncation order to consider. It should be large
			enough so that the function k -> hatL(k)+penalization(k) does not
			decrease anymore.

		Kpen: float, default=1
			Constant in front of the penalization, it has to be a positive
			number.

		alpha: float, default=1
			Regularization parameter in the Ridge regression.

		plot: boolean, default=False
			If True, plots the functions k->hatL(k), k->pen(k) and k-> hatL
			(k)+pen(k). The latter is minimized at hatm.

		Returns
		-------
		hatm: int
			The estimator of the truncation order

		objective: array, shape (max_k)
			The array of values of the objective function, minimized at hatm.
		'''
		objective=np.zeros(max_k+1)
		loss=np.zeros(max_k+1)
		pen=np.zeros(max_k+1)
		for i in range(max_k+1):
			loss[i]=self.get_hatL(Y,X,i,alpha=alpha,norm_path=False)
			pen[i]=self.get_penalization(Y.shape[0],i,Kpen=Kpen)
			objective[i]=loss[i]+pen[i]
		hatm=np.argmin(objective)

		if plot:
			plt.plot(np.arange(max_k+1),loss,label="loss")
			plt.plot(np.arange(max_k+1),pen,label="penalization")
			plt.plot(np.arange(max_k+1),objective,label="sum")
			plt.legend()
			plt.show()
		return(hatm,objective)

	def slope_heuristic(self,Kpen_values,X,Y,max_k,alpha):
		'''Implements the slope heuristic to select a value for Kpen, the
		unknown constant in front of the penalization. To this end, hatm is
		computed for several values of Kpen, and these values are then plotted
		against Kpen.

		Parameters
		----------
		Kpen_values: array, shape (n_K)
			An array of potential values for Kpen. It must be calibrated so that
			any value of hatm between 1 and max_k is obtained with values of
			Kpen in Kpen_values.

		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		Y: array, shape (n)
			Array of target values.

		max_k: int,
			Maximal value of truncation order to consider. It should be large
			enough so that the function k -> hatL(k)+penalization(k) does not
			decrease anymore.

		alpha: float, default=1
			Regularization parameter in the Ridge regression.

		Returns
		-------
		hatm: array, shape (n_K)
			The estimator hatm obtained for each value of Kpen in Kpen_values.
		'''
		hatm=np.zeros(len(Kpen_values))
		loss=np.zeros(max_k+1)
		for j in range(max_k+1):
			loss[j]=self.get_hatL(Y,X,j,alpha=alpha,norm_path=False)

		for i in range(len(Kpen_values)):
			#print(i)
			pen=np.zeros(max_k+1)
			for j in range(max_k+1):
				pen[j]=self.get_penalization(Y.shape[0],j,Kpen=Kpen_values[i])
			hatm[i]=np.argmin(loss+pen)
			#print("Hatm selected: ",hatm[i])


		# Plot
		fig, ax = plt.subplots()
		jump=1
		for i in range(max_k+1):
			if i in hatm:
				xmin=Kpen_values[hatm==i][0]
				xmax=Kpen_values[hatm==i][-1]
				ax.hlines(i,xmin,xmax,colors='b')
				if i!=0:
					ax.vlines(xmax,i,i-jump,linestyles='dashed',colors='b')
				jump=1
			else:
				jump+=1
		ax.set(xlabel=r'$K_{pen}$',ylabel=r'$\hat{m}$')
		plt.show()

		return(hatm)

		
class dataSimu(object):
	'''Object that simulated some training data.

	Parameters
	----------
	n_points: int
		Number of points in the piecewise linear approximations.

	d: int
		Dimension of the output space of the training paths.

	mast: int
		True value of the truncation order of the signature.

	Attributes
	----------
	size_sig:int
		Size of the signature truncated at mast.

	beta: array, shape (size_sig)
		True array of regression coefficients, sampled uniformly on [0,100].
	'''
	def __init__(self,n_points,d,mast):
		self.n_points=n_points
		self.d=d
		self.mast=mast
		if mast==0:
			self.size_sig=1
		else:
			self.size_sig=isig.siglength(self.d,self.mast)+1
		# self.beta=np.exp(np.arange(size_sig)/size_sig)*np.random.random(
		# 	size=size_sig)
		self.beta=100*np.random.random(size=self.size_sig)


	def get_X(self,n):
		'''Generates n sample paths X:[0,1] -> R^d, defined by
		X_t=alpha_1 + 10*alpha_2*sinus(2*pi*t/alpha_3) + 10*(t-alpha_4)^3,
		where the alphas are sampled uniformly over [0,1].X is interpolated by 
		a piecewise linear function with n_points. Each sample X is normalized
		so that it has length 1.

		Parameters
		----------
		n: int
			Number of samples to simulate.

		Returns
		-------
		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.
		'''
		X=np.zeros((n,self.n_points,self.d))
		Xlength=np.zeros((n,self.n_points,self.d))
		times=np.linspace(0,1,num=self.n_points)
		for i in range(n):
			for j in range(self.d-1):
				param=np.random.random(size=4)
				X[i,:,j]=param[0]+ 10*param[1]*np.sin(
					times*np.pi*2/param[2])+ 10*(times-param[3])**3
			X[i,:,self.d-1]=times
		return(X)

	def get_Y_sig(self,X,noise_std,plot=False):
		'''Compute the target values Y as scalar products of the truncated
		signatures of rows of X with a certain parameter beta plus a gaussian
		noise. Y follows therefore the expected signature model.
		
		Parameters
		----------
		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		noise_std: float
			Standard error of the Gaussian noise.

		plot: boolean, default=False
			If True, output two plots: one plot with the signature coefficients
			ofone sample and the regression vector beta, one scatter plot with Y
			against Y+noise to check the calibration of the noise variance.

		Returns
		-------
		Y: array, shape (n)
			Target values, defined by the expected signature model.
		'''
		n=X.shape[0]
		Y=np.zeros(n)
		noise=np.random.normal(scale=noise_std,size=n)

		SigX=get_SigX(X,self.mast)
		beta_repeated=np.repeat(self.beta.reshape(1,self.size_sig),n,axis=0)
		Y=np.sum(beta_repeated*SigX,axis=1)/1000

		if plot:
			plt.plot(SigX[0,:],label="Signature coefficients")
			plt.plot(SigX[1,:],label="Signature coefficients")
			plt.plot(self.beta,label="beta")
			plt.title("SigX and beta")
			plt.legend()
			plt.show()
			plt.plot(SigX[0,:]*self.beta,label="product")
			plt.show()
			plt.scatter(Y,Y+noise)
			plt.title("Y against Y+noise")
			plt.show()
		return(Y+noise)

	def get_Y_interaction(self,X,noise_std,plot=False):
		n=X.shape[0]
		Y=np.zeros(n)
		noise=np.random.normal(scale=noise_std,size=n)
		#Y=np.sum(X[:,:,0]*np.exp(-X[:,:,0]**2),axis=1)
		Y=np.mean(X[:,:,0]*X[:,:,1],axis=1)
		print(Y.shape)
		print(noise.shape)
		if plot:
			plt.scatter(Y,Y+noise)
			plt.title("Y against Y+noise")
			plt.show()
		return(Y+noise)

	#def get_Y_nonlinear(self,X,noise_std):
	# 	n=X.shape[0]
	# 	Y=np.zeros(n)
	# 	for i in range(n):
	# 		Y[i]+=np.mean(np.prod(X[i,:,:],axis=1))*10000
	# 	noise=np.random.normal(scale=noise_std,size=n)
	# 	#plt.scatter(Y,Y+noise)
	# 	#plt.show()
	# 	return(Y+noise)














