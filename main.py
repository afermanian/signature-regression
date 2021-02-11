from sacred import Experiment
import numpy as np
from train import SignatureRegression, BasisRegression, select_nbasis_cv, select_hatm_cv, SignatureOrderSelection
from configurations import configs
from utils import gridsearch
from tools import add_time
from get_data import get_train_test_data
import sys


ex = Experiment()

@ex.config
def my_config():
    regressor = 'signature'
    d = None
    npoints = 100
    noise_X_std = 0
    ntrain = 100
    nval = 100
    selection_method = 'cv'
    X_type = 'weather'
    Y_type = None
    nclients = None
    scaling = True
    scale_X = False
    Kpen = None

@ex.main
def my_main(_run, d, npoints, noise_X_std, ntrain, nval, regressor, selection_method, Kpen, X_type, Y_type, nclients,
            scaling, scale_X):
    try:
        Xtrain, Ytrain, Xval, Yval = get_train_test_data(X_type, ntrain=ntrain, nval=nval,  Y_type=Y_type,
                                                         npoints=npoints, d=d, noise_X_std=noise_X_std,
                                                         nclients=nclients, scale_X=scale_X)
        if regressor == 'signature':
            Xtimetrain = add_time(Xtrain)
            Xtimeval = add_time(Xval)

            if selection_method == 'cv':
                print(Xtimetrain.shape)
                hatm = select_hatm_cv(Xtimetrain, Ytrain, scaling=scaling)
            elif selection_method == 'estimation':
                order_sel = SignatureOrderSelection(Xtimetrain.shape[2], Kpen=Kpen)
                hatm = order_sel.get_hatm(Xtimetrain, Ytrain, Kpen_values=np.linspace(10 ** (-5), 10 ** (-1), num=200))
            else:
                raise NameError('selection_method not well specified')

            _run.log_scalar("hatm", hatm)
            sig_reg = SignatureRegression(hatm, scaling=scaling)
            sig_reg.fit(Xtimetrain, Ytrain)

            _run.log_scalar("val.error", sig_reg.get_loss(Xtimeval, Yval))
            _run.log_scalar("training.error", sig_reg.get_loss(Xtimetrain, Ytrain))

        elif regressor in ['bspline', 'fourier', 'fPCA']:
            nbasis = select_nbasis_cv(Xtrain, Ytrain, regressor)
            spline_reg = BasisRegression(nbasis, basis_type=regressor)
            spline_reg.fit(Xtrain, Ytrain)

            _run.log_scalar("nbasis", nbasis)
            _run.log_scalar("val.error", spline_reg.get_loss(Xval, Yval))
            _run.log_scalar("training.error", spline_reg.get_loss(Xtrain, Ytrain))

        else:
            raise NameError('regressor not well specified')
    except Exception as e:
        _run.log_scalar('error', str(e))


if __name__ == '__main__':
    config = configs[str(sys.argv[1])]
    dirname = str(sys.argv[1])
    niter = int(sys.argv[2])
    gridsearch(ex, config, dirname=dirname, niter=niter)











