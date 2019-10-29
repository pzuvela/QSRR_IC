import os
import time
import pandas as pd
import numpy as np
from scipy import optimize
from regression import regress_gbr, regress_xgbr, regress_plot
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import make_scorer
from multiprocessing import Pool
from func import add_true_mean_std, get_rmse
from iso2grad import model

fileroot = os.getcwd() + "\\data\\"
# for sys in [1, 2, 3]:
# 4 : QSRR in IC
sys = 4
# for mean in ['nomean', 'mean']:
mean = 'mean'
max_iter = 10
proc_i = 4
n_splits = 3

opt_prompt = int(input('Initiate Optimisation? (1/0)\n'))
if opt_prompt == 1:
    # rawdata = pd.read_csv('{}{}{}.csv'.format(fileroot, sys, mean))
    rawdata = pd.read_csv('{}2019-QSRR_in_IC_Part_IV_data_latest.csv'.format(fileroot))
    rawdata.drop(rawdata[rawdata['logk'] == 0.000].index, inplace=True)

    # x_data = rawdata[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
    # 'W', 'Y', 'V']]
    # ANFIS (published) model
    # x_data = rawdata[['c(KOH)', 'Alc', 'Mor29e', 'Mor31p', 'G1u', 'R3m']]
    x_data = rawdata.drop(['tR', 'logk'], axis=1)
    # y_data = rawdata[['tR / min']].values.ravel()
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

    final_values = optimize.differential_evolution(reg_objective, bounds, workers=proc_i, updating='deferred',
                                                   mutation=(1.5, 1.9), popsize=20, disp=True)
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
else:
    reg_params = {'n_estimators': 403,
                  'learning_rate': 0.22,
                  'max_depth': 3}


def model_parallel(arg_iter):

    # x_data = rawdata[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
    # 'W', 'Y', 'V']]
    x_data = rawdata.drop(['tR', 'logk'], axis=1)
    # ANFIS (published) model
    # x_data = rawdata[['c(KOH)', 'Alc', 'Mor29e', 'Mor31p', 'G1u', 'R3m']]
    # y_data = rawdata[['tR / min']].values.ravel()
    y_data = rawdata[['logk']].values.ravel()
    x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
    n_splits = 3

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

    # Load isocratic data, void times & gradient data
    grad_data = np.genfromtxt(os.getcwd() + '\\data\\grad_data.csv', delimiter=',')
    t_void = np.genfromtxt(os.getcwd() + '\\data\\t_void.csv', delimiter=',')
    iso_data = np.genfromtxt(os.getcwd() + '\\data\\iso_data.csv', delimiter=',')
    tg_total = model(reg, iso_data, t_void, grad_data, sc)
    tg_total.ravel()

    print(tg_total)
    print('Iteration #{}/{} completed'.format(arg_iter[0] + 1, max_iter))
    return rmsre_train, rmsre_test, y_data, y_hat, tg_total


if __name__ == '__main__':
    print('Initiating with {} iterations'.format(max_iter))
    start_time = time.time()

    p = Pool(processes=proc_i)
    models_final = p.map(model_parallel, zip(range(max_iter)))

    masterStats = [models_final[i][:2] for i in range(max_iter)]
    y_true = models_final[0][2]
    y_pred = [models_final[i][3] for i in range(max_iter)]

    run_time = time.time() - start_time
    print('Simulation Completed')
    print('Duration: {}\n'.format(time.strftime("%H:%M:%S", time.gmtime(run_time))))

    np.savetxt("sideQSRR/tG_IC_results_test.csv", models_final[0][4], delimiter=",")

    column = ['rmsre_train', 'rmsre_test']
    add_true_mean_std(None, pd.DataFrame(masterStats, columns=column)).to_csv(
        'sideQSRR/iteration_metrics_absrmse_sys{}{}_{}iters.csv'.format(sys, mean, max_iter), header=True)

    y_pred = pd.DataFrame(y_pred)
    add_true_mean_std(y_true, y_pred).to_csv('sideQSRR/qsrr_trprediction_absrmse_sys{}{}_{}iters.csv'
                                             .format(sys, mean, max_iter), header=True)
