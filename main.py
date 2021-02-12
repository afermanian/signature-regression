from configurations import configs
from get_data import get_train_test_data
import numpy as np
from sacred import Experiment
import sys
from tools import add_time
from train import SignatureRegression, BasisRegression, select_nbasis_cv, select_hatm_cv, SignatureOrderSelection
from utils import gridsearch


ex = Experiment()

@ex.config
def my_config():
    regressor = 'signature'
    d = None
    npoints = 100
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
def my_main(_run, d, npoints, ntrain, nval, regressor, selection_method, Kpen, X_type, Y_type, nclients, scaling, scale_X):
    """Function that runs one experiment defined in configurations.py.

    Parameters
    ----------
        _run: int
            Run ID
        d: int
            Dimension of the space of the functional covariates X, from which an output Y is learned.
        npoints: int
            Number of sampling points of the data.
        ntrain: int
            Number of training samples.
        nval: int
            Number of validation samples.
        regressor: str
            Type of regression model. Possible values are 'signature', 'bspline', 'fourier', and 'fPCA'
        selection_method: str
            If regressor is 'signature', the type of method to select hatm. Possible values are 'cv' (selection by cross
            validation) and 'estimation' (selection with the estimator hatm)
        Kpen: float
            Value of the penalization constant if regressor is 'signature' and 'selection_method' is 'estimation'.
        X_type: str
            Type of functional covariates. Possible values are 'smooth_dependent', 'smooth_independent' (for the smooth
            curves with independent or dependent coordinates), 'gaussian_processes', 'weather' (for the Canadian Weather
            dataset) and 'electricity_loads' (for the Electricity Loads dataset).
        Y_type: str
            Type of response, used only if X_type is 'smooth_dependent' or 'smooth_independent'. Possible values are
            'mean', 'max', or 'sig'.
        scaling: boolean
             Whether to scale the predictor matrix, after having computed the signature or the basis expansion, to have
             zero mean and unit variance.
        scale_X: boolean
            Whether to scale the different coordinates of X to have zero mean and unit variance. This is useful if the
            orders of magnitude of the coordinates of X are very different one from another.
    """
    try:
        Xtrain, Ytrain, Xval, Yval = get_train_test_data(X_type, ntrain=ntrain, nval=nval,  Y_type=Y_type,
                                                         npoints=npoints, d=d, scale_X=scale_X)
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











