from sacred import Experiment
import numpy as np
from simulation import DataGenerator
from train import SignatureRegression

ex = Experiment()

@ex.config
def my_config():
    d = 2
    npoints = 100
    noise_X_std = 0
    ntrain = 100
    method = 'signature'
    hatm = 2

@ex.main
def my_main(d, npoints, noise_X_std, ntrain, method, hatm):
    sim = DataGenerator(d, npoints, noise_std=noise_X_std)
    Xraw = sim.get_X(ntrain + 1)
    Y = np.mean(Xraw[:, -1, :], axis=1)
    X = Xraw[:, :-1, :]

    if method == 'signature':
        reg = SignatureRegression(d, hatm)
        reg.fit_alpha(X, Y)
        reg.fit(X, Y)
        print("Training error: ", reg.get_loss(X, Y))

if __name__ == '__main__':
    ex.run()













