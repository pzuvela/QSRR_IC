**QSRR in IC repository**

Code, examples, data & results for the paper:

Gradient retention time modelling in ion chromatoraphy through ensemble learning-powered quantitative structure-retention relationships

Package *qsrr_ic* in this repo can be used as a console application by running:

python -m qsrr_ic -f config.json

Package also exposes several APIs that can be used in scripts, notebooks or other programs. Examples are found in the *notebooks* directory.

To reproduce the results from the paper (not fully as random seeds may be different), run:

python -m qsrr_ic -f ./settings/optimization.json