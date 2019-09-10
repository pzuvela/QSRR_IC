import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize, minimize_scalar, Bounds
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
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

df = labelling(modeldata, limxc, limdelc, None, 'delc')
scaled_data, labelset, trset, sc = splitting(df, 'Sequest')

x_train, x_test, y_train, y_test = labelset


def fun(x):

    print(x)

    n_est = int(np.round(x[0], decimals=0))
    min_sam = int(np.round(x[1], decimals=0))
    lr = x[2]

    clf = GradientBoostingClassifier(n_estimators=n_est, min_samples_split=min_sam, learning_rate=lr)

    kfold = KFold(n_splits=3)
    score = cross_val_score(clf, x_train, y_train, cv=kfold)

    # clf = GradientBoostingClassifier(n_estimators=n_est, min_samples_split=min_sam, learning_rate=lr)
    # clf.fit(x_train, y_train)

    # print((1 - clf.score(x_test, y_test))**2)

    # clf model statistics
    # cm = confusion_matrix(y_train, y_testpred)
    # tn, fp, fn, tp = cm.ravel()
    # mcc = ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    print(1 - np.mean(score))

    return 1 - np.mean(score)


bounds = Bounds([100, 5, 0.001], [1000, 50, 0.1])
# initial values
lr0 = (0.1 - 0.001) / 2
n_est0 = (1000 - 100) / 2
min_sam0 = (50 - 5) / 2

initial = np.array([n_est0, min_sam0, lr0])
opt = minimize(fun, initial, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'disp': True, 'eps': 1})

# opt = minimize_scalar(fun, bounds=(0.001, 0.1), method='bounded', options={'maxiter': 500, 'disp': True})
print(int(np.round(opt.x[0], decimals=0)), int(np.round(opt.x[1], decimals=0)), opt.x[2])


