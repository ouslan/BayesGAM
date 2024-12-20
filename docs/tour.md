# A Tour of BayesGAM

## Introduction

Generalized Additive Models (GAMs) are smooth semi-parametric models of the form:

$$
    g(\mathbb{E}[y|X]) = \beta_0 + f_1(X_1) + f_2(X_2, X3) + \ldots + f_M(X_N)
$$

where `X.T = [X_1, X_2, ..., X_N]` are independent variables, `y` is the dependent variable, and `g()` is the link function that relates our predictor variables to the expected value of the dependent variable.

The feature functions `f_i()` are built using **penalized B splines**, which allow us to **automatically model non-linear relationships** without having to manually try out many different transformations on each variable.


![Basis splines](assets/pygam_basis.png)

GAMs extend generalized linear models by allowing non-linear functions of features while maintaining additivity. Since the model is additive, it is easy to examine the effect of each `X_i` on `Y` individually while holding all other predictors constant.

The result is a very flexible model, where it is easy to incorporate prior knowledge and control overfitting.

$$
y \sim ExponentialFamily(\mu|X)
$$

where 
$$
g(\mu|X) = \beta_0 + f_1(X_1) + f_2(X_2, X3) + \ldots + f_M(X_N)
$$

So we can see that a GAM has 3 components:

- ``distribution`` from the exponential family
- ``link function`` $g(\cdot)$
- ``functional form`` with an additive structure $\beta_0 + f_1(X_1) + f_2(X_2, X3) + \ldots + f_M(X_N)$

### Distribution: 
Specified via: ``GAM(distribution='...')``

Currently you can choose from the following:

- `'normal'`
- `'binomial'`
- `'poisson'`
- `'gamma'`
- `'inv_gauss'`

### Link function: 
We specify this using: ``GAM(link='...')``

Link functions take the distribution mean to the linear prediction. So far, the following are available:

- `'identity'`
- `'logit'`
- `'inverse'`
- `'log'`
- `'inverse-squared'`


### Functional Form: 
Speficied in ``GAM(terms=...)`` or more simply ``GAM(...)``

In BayesGAM, we specify the functional form using terms:

- `l()` linear terms: for terms like $X_i$
- `s()` spline terms
- `f()` factor terms
- `te()` tensor products
- `intercept`  

With these, we can quickly and compactly build models like:

```py
from pygam import PoissonGAM, s, te
from pygam.datasets import chicago

X, y = chicago(return_X_y=True)

gam = PoissonGAM(s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)
```