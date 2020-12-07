import numpy as np
import iisignature as isig
import matplotlib.pyplot as plt
import seaborn as sns

from tools import get_sigX

sns.set()


class DataGenerator(object):
	"""Generate some training data.

	Parameters
	----------
	n_points: int
		Number of points in the piecewise linear approximations.

	d: int
		Dimension of the output space of the training paths.

	noise_std: float or None
		Variance of the Gaussian noise on X
	"""
	def __init__(self, npoints, d, noise_std=0):
		self.npoints = npoints
		self.d = d
		self.noise_std = noise_std

	def get_X(self, n):
		""" Generates n sample paths X:[0,1] -> R^d, defined by
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
		"""
		X = np.zeros((n, self.npoints, self.d))

		times = np.linspace(0, 1, num=self.npoints)
		for i in range(n):
			for j in range(self.d):
				param = np.random.random(size=4)
				X[i, :, j] = param[0] + 10 * param[1] * np.sin(
					times * np.pi * 2 / param[2]) + 10 * (times - param[3]) ** 3
		noise = 2 * self.noise_std * np.random.random(size=X.shape) - self.noise_std
		return X + noise

	def get_X_dependent(self, n):

		X = np.zeros((n, self.npoints, self.d))

		times = np.linspace(0, 1, num=self.npoints)
		for i in range(n):
			sinus_param = np.random.random(size=4)
			for j in range(self.d):
				param = np.random.random() * sinus_param
				X[i, :, j] = param[0] + 10 * param[1] * np.sin(
					times * np.pi * 2 / param[2]) + 10 * (times - param[3]) ** 3
		noise = 2 * self.noise_std * np.random.random(size=X.shape) - self.noise_std
		return X + noise

	def get_Y_sig(self, X, mast, noise_std, plot=False):
		"""Compute the target values Y as scalar products of the truncated
		signatures of rows of X with a certain parameter beta plus a gaussian
		noise. Y follows therefore the expected signature model.

		Parameters
		----------
		X: array, shape (n,n_points,d)
			Array of training paths. It is a 3-dimensional array, containing
			the coordinates in R^d of n piecewise linear paths, each composed of
			n_points.

		mast: int
			True value of the truncation order of the signature.

		noise_std: float
			Amount of noise of Y, Y is equal to the scalar product of the
			signature of X against beta plus a uniform noise on
			[-noise_std,noise_std].

		plot: boolean, default=False
			If True, output two plots: one plot with the signature coefficients
			ofone sample and the regression vector beta, one scatter plot with Y
			against Y+noise to check the calibration of the noise variance.

		Returns
		-------
		Y: array, shape (n)
			Target values, defined by the expected signature model.
		"""
		n = X.shape[0]

		if mast == 0:
			size_sig = 1
		else:
			size_sig = isig.siglength(self.d, mast) + 1
		beta = np.random.random(size=size_sig) / 1000
		noise = 2 * noise_std * np.random.random(size=n) - noise_std

		SigX = get_sigX(X, mast)
		beta_repeated = np.repeat(beta.reshape(1, size_sig), n, axis=0)
		Y = np.sum(beta_repeated * SigX, axis=1)

		if plot:
			plt.scatter(Y, Y + noise)
			plt.title("Y against Y+noise")
			plt.show()
		return Y + noise

	def get_XY(self, ntrain, X_type='independent', Y_type='mean'):
		if X_type == 'dependent':
			Xraw = self.get_X_dependent(ntrain)
		else:
			Xraw = self.get_X(ntrain)
		if Y_type=='mean':
			Y = np.mean(Xraw[:, -1, :], axis=1)
		elif Y_type=='max':
			Y = np.max(Xraw[:, -1, :], axis=1)
		X = Xraw[:, :-1, :]
		return X, Y

# class dataGenerator(object):
# 	def __init__(self,nb_points=100,nb_simu=1,dim=1,noise_std=0):
# 		self.nb_points=nb_points
# 		self.nb_simu=nb_simu
# 		self.dim=dim
# 		X=np.zeros((self.nb_simu,self.nb_points,self.dim))
# 		times=np.linspace(0,1,num=self.nb_points)
# 		for i in range(self.nb_simu):
# 			for j in range(dim):
# 				param=np.random.random(size=4)
# 				X[i,:,j]=param[0]+ 10*param[1]*np.sin(times*np.pi*2/param[1]) + 10*(times-param[2])**3
# 		self.data=X
# 		self.data_noise=X+np.random.normal(scale=noise_std,size=np.shape(X))
#
# 	def get_output(self,model,noise_std=0.1,task="regression"):
# 		Y=np.zeros(self.nb_simu)
# 		noise=np.random.normal(scale=noise_std,size=self.nb_simu)
# 		deltat=1/self.nb_points
# 		beta=np.cos(np.linspace(0,1,num=self.nb_points)*2*np.pi)
# 		if model=="linear":
# 			for i in range(self.nb_simu):
# 				for j in range(self.dim):
# 					Y[i]+=np.sum(self.data[i,1:,j]*beta[1:] - self.data[i,:-1,j]*beta[:-1])*deltat/self.dim
#
# 		elif model=="nonlinear":
# 			for i in range(self.nb_simu):
# 				for j in range(self.dim):
# 					Y[i]+=np.sum(beta[1:]*(self.data[i,1:,j])**2 -beta[:-1] *(self.data[i,:-1,j])**2)*deltat/self.dim
#
# 		elif model=="interaction":
# 			if self.dim==1:
# 				raise Exception("Model with interaction needs at least 2 dimensions")
# 			for i in range(self.nb_simu):
# 				for j in range(self.dim-1):
# 					Y[i]+=np.sum(((self.data[i,1:,j])**2)*np.log(np.abs(self.data[i,1:,j+1])) -
# 						((self.data[i,:-1,j])**2)*np.log(np.abs(self.data[i,:-1,j+1])))*deltat/(self.dim-1)
# 		elif model=="sparse":
# 			for i in range(self.nb_simu):
# 				Y[i]=np.sum(beta[1:]*(self.data[i,1:,0])**2 -beta[:-1] *(self.data[i,:-1,0])**2)*deltat/self.dim
# 		elif model=="signature":
# 			sig=signatureTransform(2,self.dim)
# 			Y=np.sum(sig.get_signature(self.data),axis=1)
# 		else:
# 			Y=np.arange(self.nb_simu)
#
#
# 		if task=="regression":
# 			Y_noise=(Y-np.mean(Y))/np.std(Y)+noise
# 			#plt.scatter(Y,Y_noise)
# 			#plt.show()
# 			return(Y_noise)
# 		else:
# 			Y_noise=Y+noise
# 			return(Y_noise >=0.5)