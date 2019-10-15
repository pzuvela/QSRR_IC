import os
import time
import pandas as pd
import numpy as np
import scipy.optimize as opt
import func
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from preprocessing import labelling, splitting

"""
hyper parameter optimisation for:
1) n_estimators
2) learning rate
3) max_depth
"""
modeldata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
limxc = [1.9, 2.2, 3.75]
limdelc = 0.08
n_splits = 5

df = labelling(modeldata, limxc, limdelc, None, 'delc')
scaled_data, labelset, trset, sc, tr_max = splitting(df, 'Sequest')
x_train, x_test, y_train, y_test = trset

# Optimisation function begins here, takes in x_train, x_test, y_train, y_test
# Initial values
model = XGBRegressor(objective="reg:squarederror").fit(x_train, y_train)
initial = model.get_params()

y_hat = model.predict(x_train)
initial_rmsre_train = func.get_rmsre(y_train, y_hat)

y_hat = model.predict(x_test)
initial_rmsre_test = func.get_rmsre(y_test, y_hat)

start_time = time.time()

print('    ---------------- Initial Parameters ---------------')
print('    n_estimators: {:.0f}\n'
      '    learning_rate: {:.2f} \n'
      '    max_depth: {:.0f}\n'
      '    Initial %RMSEE: {:.2f}'
      '    Initial %RMSEP: {:.2f}'
      .format(initial['n_estimators'],
              initial['learning_rate'],
              initial['max_depth'],
              initial_rmsre_train,
              initial_rmsre_test
              )
      )
print('    ---------------------------------------------------')


# Creating optimisation function
def objective(x):
    # Descaling Parameters
    n_est = int(np.round(x[0], decimals=0))
    lr = x[1]
    max_depth = int(np.round(x[2], decimals=0))

    opt_model = model.set_params(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)

    # CV score
    scorer = make_scorer(func.get_rmsre)
    score = cross_val_score(opt_model, x_train, y_train, cv=KFold(n_splits=n_splits), scoring=scorer)

    return np.mean(score)


# Creating bounds
n_est_min, n_est_max = 10, 500
lr_min, lr_max = 0.01, 0.9
max_depth_min, max_depth_max = 1, 5
bounds = opt.Bounds([n_est_min, lr_min, max_depth_min],
                    [n_est_max, lr_max, max_depth_max])

final_values = opt.differential_evolution(objective, bounds, workers=-1, updating='deferred',
                                          mutation=(1.5, 1.9), popsize=20)
opt_dict = {'n_estimators': int(np.round(final_values.x[0], decimals=0)),
            'learning_rate': final_values.x[1],
            'max_depth': int(np.round(final_values.x[2], decimals=0))
            }

model.set_params(**opt_dict).fit(x_train, y_train)

y_hat = model.predict(x_train)
final_rmsre_train = func.get_rmsre(y_train, y_hat)

y_hat = model.predict(x_test)
final_rmsre_test = func.get_rmsre(y_test, y_hat)

elapsed_time = time.time() - start_time


print('    ----------------- Final Parameters ----------------')
print('    n_estimators: {:.0f}\n'
      '    learning_rate: {:.2f} \n'
      '    max_depth: {:.0f}\n'
      '    Final %RMSECV: {:.3f}\n'
      '    Final %RMSEE: {:.2f}'
      '    Final %RMSEP: {:.2f}\n'
      '    Optimisation Duration: {}'
      .format(int(np.round(final_values.x[0], decimals=0)),
              final_values.x[1],
              int(np.round(final_values.x[2], decimals=0)),
              final_values.objective,
              final_rmsre_train,
              final_rmsre_test,
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
              )
      )
print('    ---------------------------------------------------')
# just model().set_params(**opt_dict).fit(x_train, y_train to apply to the model
# unable to make it a function due to how multiprocessing works.
