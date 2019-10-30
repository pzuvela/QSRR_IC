import os
import pandas as pd
import numpy as np
from regression import regress_gbr, regress_xgbr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from func import add_true_mean_std, get_rmse
from iso2grad import model
from matplotlib import pyplot as plt

reg_params = {'n_estimators': 403,
              'learning_rate': 0.22,
              'max_depth': 3}

rawdata = pd.read_csv(os.getcwd() + '/data/2019-QSRR_in_IC_Part_IV_data_latest.csv')
x_data = rawdata.drop(['tR', 'logk'], axis=1)
y_data = rawdata[['logk']].values.ravel()
x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)

sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train_unscaled), columns=x_train_unscaled.columns).values
x_test = pd.DataFrame(sc.transform(x_test_unscaled), columns=x_test_unscaled.columns).values
x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns).values
trset = [x_train, x_test, y_train, y_test]

# GBR
# reg, reg_traindata, reg_testdata = regress_gbr(trset, reg_params=None)

# XGB
reg, reg_traindata, reg_testdata = regress_xgbr(trset, reg_params=reg_params)

y_hat_train = reg.predict(x_train)
rmsre_train = get_rmse(y_train, y_hat_train)
y_hat_test = reg.predict(x_test)
rmsre_test = get_rmse(y_test, y_hat_test)
y_hat = reg.predict(x_data)

# Gradient profiles
grad_data = np.genfromtxt(os.getcwd() + '/data/grad_data.csv', delimiter=',')

# Void times
t_void = np.genfromtxt(os.getcwd() + '/data/t_void.csv', delimiter=',')

# Isocratic data for all the analytes
iso_data = np.genfromtxt(os.getcwd() + '/data/iso_data.csv', delimiter=',')

# Gradient retention times
tg_exp = np.genfromtxt(os.getcwd() + '/data/tg_data.csv', delimiter=',')

tg_total = model(reg, iso_data, t_void, grad_data, sc)

tg = tg_total.flatten(order='F')

fig1, ax1 = plt.subplots()
ax1.scatter(tg_exp, tg, alpha=0.7, c='C0')

fig2, ax2 = plt.subplots()
ax2.scatter(y_data, y_hat, alpha=0.7, c='C0')

plt.show()
