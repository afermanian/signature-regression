# Linear functional regression with truncated signatures [preprint](arxiv)

We propose a novel methodology for regressing a real output on vector-valued functional covariates. This methodology is based on the notion of signature, which is a representation of a function as an infinite series of its iterated integrals. The signature depends crucially on a truncation parameter for which an estimator is provided, together with theoretical guarantees. We provide here the code to compute an estimator of the truncation parameter, which is then used to implement a linear regression with signature features. The procedure is summarised below:

<p align="center">
    <img class="center" src="./paper/images/algo-full-procedure.png" width="500"/>
</p>

## The code

All the code to fit a linear regression on signature features is in `main.py`. Given data matrices Y of size n and X of size n x p x d, the following lines of code will give you the estimator of the truncation order and a vector of regression coefficients:

`from main import orderEstimator

est=orderEstimator(d)
hatm=est.get_hatm(Y,X,M,Kpen=Kpen,alpha=alpha)[0]
reg,Ypred=est.fit_ridge(Y,X,hatm,alpha=alpha,norm_path=False)
`

where alpha is the regularization parameter, Kpen is the constant in the penalization of the truncation order and M is the range of truncation orders considered.

## Citation


## Reproducing the experiments

### Environment

### Data

### Running the scripts




