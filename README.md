# Linear functional regression with truncated signatures ([preprint](https://arxiv.org/abs/2006.08442))

We propose a novel methodology for regressing a real output on vector-valued functional covariates. This methodology is based on the notion of signature, which is a representation of a function as an infinite series of its iterated integrals. The signature depends crucially on a truncation parameter for which an estimator is provided, together with theoretical guarantees. We provide here the code to compute an estimator of the truncation parameter, which is then used to implement a linear regression with signature features. The procedure is summarised below:

<p align="center">
    <img class="center" src="./paper/images/algo-full-procedure.png" width="600"/>
</p>

## The code

All the code to fit a linear regression on signature features is in `main.py`. Given data matrices Y of size n and X of size n x p x d, the following lines of code will give you the estimator of the truncation order and a vector of regression coefficients:

```python
from main import orderEstimator

est=orderEstimator(d)
hatm=est.get_hatm(Y,X,M,Kpen=Kpen,alpha=alpha)[0]
reg,Ypred=est.fit_ridge(Y,X,hatm,alpha=alpha,norm_path=False)
```

where alpha is the regularization parameter, Kpen is the constant in the penalization of the truncation order estimator and M is the range of truncation orders considered. If you wish to calibrate Kpen with the slope heuristics method, the command

```python
K_values=np.linspace(10**(-7),10**1,num=200)
est.slope_heuristic(K_values,X,Y,max_k,alpha)
```
will output a plot of the estimator as a function of the constant. Then, you can choose the best constant by looking at the biggest jump and picking twice the x-value corresponding to this jump. 

## Reproducing the experiments

We give below the steps to reproduce the results of the paper.

### Environment

All the necessary packages may be set up by running
`pip install -r requirements.txt`

### Data

First create a directory in the root directory `data/`. Then download the Canadian Weather and UCR & UEA datasets from the following sources:

* The Canadian Weather dataset comes from the `fda` R package and can be downloaded by running in R the scrip `get_data/get_canadian_weather.R`
* The UCR & UEA datasets may be downloaded at http://www.timeseriesclassification.com and should all be stored in `data/ucr/` as .arff files.

### Running the scripts

There are 3 scripts that reproduce the various experiments of the paper.

* Run `python script_cvg_hatm_Y_sig.py` to get the results on simulated datasets. Beware that this script may take a while to run. Its results are stored in `results/`, together with the corresponding plots.
* Run `python script_canadian_weather.py` to get the results on the Canadian Weather dataset. You will be asked to enter the constant of the penalization chosen by the slope heuristics method.
* Run `python script_ucr.py <name>` where `<name>` should be the name of one of the UCR & UEA datasets to reproduce the results on these datasets.

## Citation

@article{fermanian2020linear,
  title={Linear functional regression with truncated signatures},
  author={Fermanian, Adeline},
  journal={arXiv:2006.08442},
  year={2020}
}


