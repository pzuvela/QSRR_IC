"""
QSRR model development in IC

Required packages:
1) Pandas
2) Numpy
3) Scipy
4) xgBoost
5) scikit-learn

"""
import os
import time
import pandas as pd
import numpy as np
from scipy import optimize
from regression import regress_xgbr
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import make_scorer
from multiprocessing import Pool
from func import add_true_mean_std, get_rmse
from iso2grad import model

# Root data directory
fileroot = os.getcwd() + '/data/'

# Fixed variables
max_iter = 100
proc_i = 4
n_splits = 3

opt_prompt = str(input('Initiate Optimisation (default: no) ? (yes / no) '))

if opt_prompt == 'yes':

    rawdata = pd.read_csv(os.getcwd() + '/data/2019-QSRR_in_IC_Part_IV_data_latest.csv')
    rawdata.drop(rawdata[rawdata['logk'] == 0.000].index, inplace=True)

    x_data = rawdata.drop(['tR', 'logk'], axis=1)
    y_data = rawdata[['logk']].values.ravel()
    x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,
                                                                          shuffle=True)

    sc = StandardScaler()
    x_train = pd.DataFrame(sc.fit_transform(x_train_unscaled), columns=x_train_unscaled.columns).values
    x_test = pd.DataFrame(sc.transform(x_test_unscaled), columns=x_test_unscaled.columns).values
    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns).values

    reg = XGBRegressor(objective="reg:squarederror").fit(x_train, y_train)

    # QSRR Optimisation
    print('    ----------------- QSRR Optimisation ---------------')

    # Initial Params
    initial = reg.get_params()
    y_hat = reg.predict(x_train)
    initial_rmse_train = get_rmse(y_train, y_hat)
    y_hat = reg.predict(x_test)
    initial_rmse_test = get_rmse(y_test, y_hat)
    regopt_start = time.time()

    toprint = '    ---------------- Initial Parameters ---------------\n' \
              '    n_estimators: {:.0f}\n' \
              '    learning_rate: {:.2f} \n' \
              '    max_depth: {:.0f}\n' \
              '    Initial RMSEE: {:.2f}' \
              '    Initial RMSEP: {:.2f}\n' \
              '    ---------------------------------------------------' \
        .format(initial['n_estimators'],
                initial['learning_rate'],
                initial['max_depth'],
                initial_rmse_train,
                initial_rmse_test
                )

    print(toprint)

    # Creating bounds
    n_est_min, n_est_max = 10, 500
    lr_min, lr_max = 0.01, 0.9
    max_depth_min, max_depth_max = 1, 5
    bounds = optimize.Bounds([n_est_min, lr_min, max_depth_min],
                             [n_est_max, lr_max, max_depth_max])

    # Creating optimisation function, needs to be in each
    def reg_objective(x):
        # Descaling Parameters
        n_est = int(np.round(x[0], decimals=0))
        lr = x[1]
        max_depth = int(np.round(x[2], decimals=0))

        opt_model = reg.set_params(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)

        # CV score
        scorer = make_scorer(get_rmse)
        score = cross_val_score(opt_model, x_train, y_train, cv=KFold(n_splits=n_splits), scoring=scorer)

        return np.mean(score)

    final_values = optimize.differential_evolution(reg_objective, bounds, workers=-1, updating='deferred',
                                                   mutation=(1.5, 1.9), popsize=20)
    reg_params = {'n_estimators': int(np.round(final_values.x[0], decimals=0)),
                  'learning_rate': final_values.x[1],
                  'max_depth': int(np.round(final_values.x[2], decimals=0))}
    reg.set_params(**reg_params).fit(x_train, y_train)

    # Final Params
    y_hat = reg.predict(x_train)
    final_rmse_train = get_rmse(y_train, y_hat)
    y_hat = reg.predict(x_test)
    final_rmse_test = get_rmse(y_test, y_hat)
    regopt_time = time.time() - regopt_start

    toprint = '    ----------------- Final Parameters ----------------\n' \
              '    n_estimators: {:.0f}\n' \
              '    learning_rate: {:.2f} \n' \
              '    max_depth: {:.0f}\n' \
              '    Final RMSECV: {:.3f}\n' \
              '    Final RMSEE: {:.2f}' \
              '    Final RMSEP: {:.2f}\n' \
              '    Optimisation Duration: {}\n' \
              '    ---------------------------------------------------\n' \
        .format(int(np.round(final_values.x[0], decimals=0)),
                final_values.x[1],
                int(np.round(final_values.x[2], decimals=0)),
                final_values.fun,
                final_rmse_train,
                final_rmse_test,
                time.strftime("%H:%M:%S", time.gmtime(regopt_time))
                )
    print(toprint)

    with open(os.getcwd() + '/sideQSRR/2019-QSRR_IC_PartIV-opt.txt', "w") as text_file:
        text_file.write(toprint)

else:
    reg_params = {'n_estimators': 403,
                  'learning_rate': 0.22,
                  'max_depth': 3}


def model_parallel(arg_iter):

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
    t_void = np.genfromtxt(os.getcwd() + '/data/t_void_test3.csv', delimiter=',')

    # Isocratic data for all the analytes+
    iso_data = np.genfromtxt(os.getcwd() + '/data/iso_data_test3.csv', delimiter=',')

    # Gradient retention times
    tg_exp = np.genfromtxt(os.getcwd() + '/data/tg_data_test3.csv', delimiter=',')

    # Predicted retention times
    tg_total = model(reg, iso_data, t_void, grad_data, sc).flatten(order='F')

    rmsre_grad = get_rmse(tg_exp, tg_total)

    print('Iteration #{}/{} completed'.format(arg_iter[0] + 1, max_iter))
    return rmsre_train, rmsre_test, y_data, y_hat, rmsre_grad, tg_exp, tg_total


if __name__ == '__main__':

    # Display initialization and initialize start time
    print('Initiating with {} iterations'.format(max_iter))
    start_time = time.time()

    # Start Parallel Pool with "proc_i" processes
    p = Pool(processes=proc_i)
    models_final = p.map(model_parallel, zip(range(max_iter)))

    # Training and testing RMSE
    rmse_iso = [models_final[i][:2] for i in range(max_iter)]

    # True and predicted isocratic retention times
    y_true = models_final[0][2]
    y_pred = [models_final[i][3] for i in range(max_iter)]

    # Gradient RMSE
    rmse_grad = [models_final[i][4] for i in range(max_iter)]

    # True and predicted gradient retention times
    tg_true = models_final[0][5]
    tg_pred = [models_final[i][6] for i in range(max_iter)]

    # Compute and display run-time
    run_time = time.time() - start_time
    print('Simulation Completed')
    print('Duration: {}\n'.format(time.strftime("%H:%M:%S", time.gmtime(run_time))))

    # Save the distribution of isocratic retention time errors
    column = ['rmsre_train', 'rmsre_test']
    add_true_mean_std(None, pd.DataFrame(rmse_iso, columns=column)).to_csv(
        os.getcwd() + '/sideQSRR/2019-QSRR_IC_PartIV-errors_iso_{}_iters.csv'.format(max_iter), header=True)

    # Save predicted isocratic retention times
    y_pred = pd.DataFrame(y_pred)
    add_true_mean_std(y_true, y_pred).to_csv(os.getcwd() + "/sideQSRR/2019-QSRR_IC_PartIV-tR_iso_{}_iters.csv"
                                             .format(max_iter), header=True)

    # Save the distribution of gradient retention time errors
    column = ['rmsre_grad']
    add_true_mean_std(None, pd.DataFrame(rmse_grad, columns=column)).to_csv(
        os.getcwd() + '/sideQSRR/2019-QSRR_IC_PartIV-errors_grad_{}_iters.csv'.format(max_iter),
        header=True)

    # Save predicted gradient retention times
    tg_pred = pd.DataFrame(tg_pred)
    add_true_mean_std(y_true, y_pred).to_csv(os.getcwd() + "/sideQSRR/2019-QSRR_IC_PartIV-tR_grad_{}_iters.csv"
                                             .format(max_iter), header=True)
