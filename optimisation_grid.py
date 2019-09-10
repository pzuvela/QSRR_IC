import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize, minimize_scalar, Bounds
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from preprocessing import labelling, splitting

"""
hyper parameter optimisation 
parameter = [start, end, incr]
learning_rate = [0.001. 0.1. 0.001] points = 90
n_estimators = [100, 1000, 10] points = 90
min_samples_split = [5, 50, 5] points = 9
max_depth = [1, 10, 1] points = 9 
"""

modeldata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')
limxc = [1.9, 2.2, 3.75]
limdelc = 0.08

df = labelling(modeldata, limxc, limdelc, None, 'delc')
scaled_data, labelset, trset, sc = splitting(df, 'Sequest')

x_train, x_test, y_train, y_test = labelset

# two layered grid search
roughstep = 2
finestep = 5
clf = GradientBoostingClassifier()
kf = KFold(n_splits=3)


def optimise(step):
    param_grid = {'learning_rate': np.linspace(0.001, 0.1, step),
                  'n_estimators': np.linspace(10, 100, step, dtype='int'),
                  'min_samples_split': np.linspace(5, 50, step, dtype='int'),
                  'max_depth': np.linspace(1, 10, step, dtype='int')
                  }
    clf_opt = GridSearchCV(clf, cv=kf, param_grid=param_grid)
    clf_opt.fit(x_train, y_train)
    y_hat = clf_opt.predict(x_test)
    mcc = matthews_corrcoef(y_test, y_hat)

    print(clf_opt.best_params_)
    print(mcc)
    return clf_opt.best_params_

optimise(finestep)
