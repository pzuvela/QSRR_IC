import os
import time

import pandas as pd
import numpy as np
import scipy.optimize as opt
import func
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from preprocessing import labelling, splitting

"""
hyper parameter optimisation 
parameter = [start, end, incr]
learning_rate = [0.01. 0.1. 0.01]
n_estimators = [100, 1000, 100] 
min_samples_split = [5, 50, 5]
max_depth = [1, 5, 1]
"""
modeldata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
limxc = [1.9, 2.2, 3.75]
limdelc = 0.08
n_splits = 3

df = labelling(modeldata, limxc, limdelc, None, 'delc')
scaled_data, labelset, trset, sc, tr_max = splitting(df, 'Sequest')

x_train, x_test, y_train, y_test = trset


def fun(x):
    # Descaling Parameters
    n_est = int(np.round(x[0], decimals=0))
    lr = x[1]
    max_depth = int(np.round(x[2], decimals=0))

    opt_gbr = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)

    scorer = make_scorer(mean_squared_error)

    # CV score
    score = cross_val_score(opt_gbr, x_train, y_train, cv=KFold(n_splits=n_splits), scoring=scorer)

    return np.mean(score)


# Optimisation function begins here, takes in x_train, x_test, y_train, y_test
# Initial values
start_time = time.time()
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_hat = gbr.predict(x_test)
initial = gbr.get_params()
initial_mse = mean_squared_error(y_test, y_hat)
initial_rmsre = func.get_rmsre(y_test, y_hat)

print('    ---------------- Initial Parameters ---------------')
print('    n_estimators: {:.0f}\n'
      '    learning_rate: {:.2f} \n'
      '    max_depth: {:.0f}\n'
      '    Initial MSEP: {:.2f}\n'
      '    Initial %RMSEP: {:.2f}'
      .format(initial['n_estimators'],
              initial['learning_rate'],
              initial['max_depth'],
              initial_mse,
              initial_rmsre
              )
      )
print('    ---------------------------------------------------')

# Creating bounds
n_est_min, n_est_max = 10, 500
lr_min, lr_max = 0.01, 0.9
max_depth_min, max_depth_max = 1, 5
bounds = opt.Bounds([n_est_min, lr_min, max_depth_min],
                    [n_est_max, lr_max, max_depth_max])

final_values = opt.differential_evolution(fun, bounds, disp=True)
opt_dict = {'n_estimators': int(np.round(final_values.x[0], decimals=0)),
            'learning_rate': final_values.x[1],
            'max_depth': int(np.round(final_values.x[2], decimals=0))
            }

gbr.set_params(**opt_dict)
y_hat = gbr.predict(x_test)
final_mse = mean_squared_error(y_test, y_hat)
final_rmsre = func.get_rmsre(y_test, y_hat)
elapsed_time = time.time() - start_time

print('    ----------------- Final Parameters ----------------')
print('    n_estimators: {:.0f}\n'
      '    learning_rate: {:.2f} \n'
      '    max_depth: {:.0f}\n'
      '    Final MSECV: {:.3f}\n'
      '    Final MSEP: {:.2f}'
      '    Final %RMSEP: {:.2f}\n'
      '    Optimisation Duration: {}'
      .format(int(np.round(final_values.x[0], decimals=0)),
              final_values.x[1],
              int(np.round(final_values.x[2], decimals=0)),
              final_values.fun,
              final_mse,
              final_rmsre,
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
              )
      )
print('    ---------------------------------------------------')

# return opt_dict
# just model.set_params(**opt_dict) from this for others
