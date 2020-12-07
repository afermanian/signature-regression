from sacred import Experiment
import numpy as np
from simulation import DataGenerator
from train import SignatureRegression, SplineRegression, select_nbasis_cv, select_hatm_cv, SignatureOrderSelection
from configurations import configs
from utils import gridsearch
from tools import add_time
import sys


ex = Experiment()

@ex.config
def my_config():
    regressor = 'signature'
    d = 2
    npoints = 100
    noise_X_std = 0
    ntrain = 100
    nval = 100
    selection_method = 'cv'
    X_type = 'independent'
    Y_type = 'mean'

@ex.main
def my_main(_run, d, npoints, noise_X_std, ntrain, nval, regressor, selection_method, X_type, Y_type):
    try:
        sim = DataGenerator(npoints+1, d, noise_std=noise_X_std)
        Xtrain, Ytrain = sim.get_XY(ntrain, X_type=X_type, Y_type=Y_type)
        Xval, Yval = sim.get_XY(nval)

        if regressor == 'signature':
            Xtimetrain = add_time(Xtrain)
            Xtimeval = add_time(Xval)

            if selection_method == 'cv':
                print(Xtimetrain.shape)
                hatm = select_hatm_cv(Xtimetrain, Ytrain)
            else:
                order_sel = SignatureOrderSelection(d+1)
                hatm = order_sel.get_hatm(Xtimetrain, Ytrain, Kpen_values=np.linspace(10 ** (-5), 5 *10 ** 1, num=200))

            _run.log_scalar("hatm", hatm)
            sig_reg = SignatureRegression(d, hatm)
            sig_reg.fit(Xtimetrain, Ytrain)

            _run.log_scalar("val.error", sig_reg.get_loss(Xtimeval, Yval))
            _run.log_scalar("training.error", sig_reg.get_loss(Xtimetrain, Ytrain))

        elif regressor == 'spline':
            nbasis = select_nbasis_cv(Xtrain, Ytrain)
            spline_reg = SplineRegression(nbasis)
            spline_reg.fit(Xtrain, Ytrain)

            _run.log_scalar("nbasis", nbasis)
            _run.log_scalar("val.error", spline_reg.get_loss(Xval, Yval))
            _run.log_scalar("training.error", spline_reg.get_loss(Xtrain, Ytrain))

    except Exception as e:
        _run.log_scalar('error', str(e))


if __name__ == '__main__':
    config = configs[str(sys.argv[1])]
    dirname = str(sys.argv[1])
    niter = int(sys.argv[2])
    gridsearch(ex, config, dirname=dirname, niter=niter)











