"""

QSRR model development in IC Part IV project

Required packages:
1) Pandas
2) Numpy
3) Scipy
4) xgBoost
5) scikit-learn

Usage:
main.py max_iter count proc_i method opt_prompt n_splits (opt)

"""
from os import getcwd
from sys import argv
from time import time, strftime, gmtime
from datetime import datetime
from pandas import read_csv, DataFrame
from numpy import genfromtxt, round, mean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import make_scorer
from multiprocessing import Pool
from func import get_rmse
from iso2grad import model

# Directories
curr_dir = getcwd()
data_dir = curr_dir + '/data/'
results_dir = curr_dir + '/results/'

""" Fixed variables 

Input arguments:
1) max_iter     : number of iterations
2) count        : run count (necessary if some parallel runs finish at the same time)
3) proc_i       : number of processes
4) method       : regression method (currently implemented: xgb, gbr)
5) opt_prompt   : prompt for optimization of hyper-parameters (yes, no, default: no)
6) n_splits     : number of cross-validation splits (for optimization, if opt_prompt is no, then n_splits is [])

"""

if not len(argv) > 1:
    exit('Usage: python main.py max_iter count proc_i method  opt_prompt n_splits')

max_iter = int(argv[1])
count = int(argv[2])
proc_i = int(argv[3])
method = str(argv[4])
opt_prompt = str(argv[5])

if opt_prompt == "yes":
    n_splits = int(argv[6])
else:
    n_splits = []

""" 
Loading data into Numpy arrays:
1) IC data for QSRR model building (at all experimental isocratic concentrations) 
2) Gradient profiles
3) Void times
4) Isocratic data for all the analytes (without c(eluent), that is approximated using the iso2grad model)
5) Experimental gradient retention times
"""
# IC data for QSRR
raw_data = read_csv(data_dir + '2019-QSRR_in_IC_Part_IV_data_latest.csv')

# Gradient profiles
grad_data = genfromtxt(data_dir + 'grad_data.csv', delimiter=',')

# Void times
t_void = genfromtxt(data_dir + 't_void.csv', delimiter=',')

# Isocratic data for all the analytes
iso_data = genfromtxt(data_dir + 'iso_data.csv', delimiter=',')

# Gradient retention times
tg_exp = genfromtxt(data_dir + 'tg_data.csv', delimiter=',')

""" Data processing """
# Drop rows with logk values of 0.000 (relative errors cannot be computed for these)
raw_data.drop(raw_data[raw_data['logk'] == 0.000].index, inplace=True)

# Drop retention times and logk values to generate the x_data matrix
x_data = raw_data.drop(['tR', 'logk'], axis=1)

# Define y_data and ravel it into a column/row vector
y_data = raw_data[['logk']].values.ravel()

# Hyper-parameter optimization
if opt_prompt == 'yes':

    # Randomly split the data into training and testing
    x_train_unscaled_opt, x_test_unscaled_opt, y_train_opt, y_test_opt = train_test_split(x_data, y_data,
                                                                                          test_size=0.3,
                                                                                          shuffle=True)

    # Define a scaling object
    sc_opt = StandardScaler()

    # Scale the training data and save the mean & std into the object "sc"
    x_train_opt = DataFrame(sc_opt.fit_transform(x_train_unscaled_opt), columns=x_test_unscaled_opt.columns).values

    # Scale the testing data (using the training mean & std)
    x_test_opt = DataFrame(sc_opt.transform(x_test_unscaled_opt), columns=x_test_unscaled_opt.columns).values

    # Scale all of the data (using the training mean & std)
    x_data_opt = DataFrame(sc_opt.transform(x_data), columns=x_data.columns).values

    # Import scipy optimize
    from scipy import optimize

    # xgBoost
    if method == 'xgb':

        from xgboost import XGBRegressor

        reg_opt = XGBRegressor(objective="reg:squarederror").fit(x_train_opt, y_train_opt)

    # sklearn gradient boosting
    elif method == 'gbr':

        from sklearn.ensemble import GradientBoostingRegressor

        reg_opt = GradientBoostingRegressor()
        reg_opt.fit(x_train_opt, y_train_opt)

    # Default
    else:
        from xgboost import XGBRegressor

        reg_opt = XGBRegressor(objective="reg:squarederror").fit(x_train_opt, y_train_opt)

    # QSRR Optimisation
    print('    ----------------- QSRR Optimisation ---------------')

    # Initial Params
    initial = reg_opt.get_params()
    y_hat_opt = reg_opt.predict(x_train_opt)
    initial_rmse_train = get_rmse(y_train_opt, y_hat_opt)
    y_hat_opt = reg_opt.predict(x_test_opt)
    initial_rmse_test = get_rmse(y_test_opt, y_hat_opt)
    regopt_start = time()

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
        n_est = int(round(x[0], decimals=0))
        lr = x[1]
        max_depth = int(round(x[2], decimals=0))

        opt_model = reg_opt.set_params(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)

        # CV score
        scorer = make_scorer(get_rmse)
        score = cross_val_score(opt_model, x_train_opt, y_train_opt, cv=KFold(n_splits=n_splits), scoring=scorer)

        return mean(score)


    final_values = optimize.differential_evolution(reg_objective, bounds, workers=proc_i, updating='deferred',
                                                   mutation=(1.5, 1.9), popsize=20)
    reg_params = {'n_estimators': int(round(final_values.x[0], decimals=0)),
                  'learning_rate': final_values.x[1],
                  'max_depth': int(round(final_values.x[2], decimals=0))}
    reg_opt.set_params(**reg_params).fit(x_train_opt, y_train_opt)

    # Final Params
    y_hat = reg_opt.predict(x_train_opt)
    final_rmse_train = get_rmse(y_train_opt, y_hat)
    y_hat = reg_opt.predict(x_test_opt)
    final_rmse_test = get_rmse(y_test_opt, y_hat)
    regopt_time = time() - regopt_start

    toprint = '    ----------------- Final Parameters ----------------\n' \
              '    n_estimators: {:.0f}\n' \
              '    learning_rate: {:.2f} \n' \
              '    max_depth: {:.0f}\n' \
              '    Final RMSECV: {:.3f}\n' \
              '    Final RMSEE: {:.2f}' \
              '    Final RMSEP: {:.2f}\n' \
              '    Optimisation Duration: {}\n' \
              '    ---------------------------------------------------\n' \
        .format(int(round(final_values.x[0], decimals=0)),
                final_values.x[1],
                int(round(final_values.x[2], decimals=0)),
                final_values.fun,
                final_rmse_train,
                final_rmse_test,
                strftime("%H:%M:%S", gmtime(regopt_time))
                )
    print(toprint)

    with open(results_dir + '2019-QSRR_IC_PartIV-{}_{}_opt_run_{}.txt'.format(
            datetime.now().strftime('%d_%m_%Y-%H_%M'), method, count), "w") as text_file:
        text_file.write(toprint)

else:

    # xgBoost parameters // optimized using 3-fold CV
    if method == 'xgb':

        reg_params = {'n_estimators': 497,
                      'learning_rate': 0.23,
                      'max_depth': 2}

    # GBR (sklearn) parameters // optimized using 3-fold CV
    elif method == 'gbr':

        reg_params = {'n_estimators': 485,
                      'learning_rate': 0.23,
                      'max_depth': 2}

    # Default
    else:

        reg_params = {'n_estimators': 497,
                      'learning_rate': 0.23,
                      'max_depth': 2}


# Defining a function to feed to multiprocessing
def model_parallel(arg_iter):

    # Randomly split the data into training and testing
    x_train_unscaled_par, x_test_unscaled_par, y_train_par, y_test_par = train_test_split(x_data, y_data, test_size=0.3,
                                                                                          shuffle=True)

    # Define a scaling object
    sc_par = StandardScaler()

    # Scale the training data and save the mean & std into the object "sc"
    x_train_par = DataFrame(sc_par.fit_transform(x_train_unscaled_par), columns=x_train_unscaled_par.columns).values

    # Scale the testing data (using the training mean & std)
    x_test_par = DataFrame(sc_par.transform(x_test_unscaled_par), columns=x_test_unscaled_par.columns).values

    # Scale all of the data (using the training mean & std)
    x_data_par = DataFrame(sc_par.transform(x_data), columns=x_data.columns).values

    trset = [x_train_par, x_test_par, y_train_par, y_test_par]

    # XGB
    if method == 'xgb':
        from regr import regress_xgbr
        reg, _, _ = regress_xgbr(trset, reg_params=reg_params)

    # GBR
    elif method == 'gbr':
        from regr import regress_gbr
        reg, _, _ = regress_gbr(trset, reg_params=reg_params)

    # Default
    else:
        from regr import regress_xgbr
        reg, _, _ = regress_xgbr(trset, reg_params=reg_params)

    y_hat_train_par = reg.predict(x_train_par)
    rmsre_train_par = get_rmse(y_train_par, y_hat_train_par)
    y_hat_test_par = reg.predict(x_test_par)
    rmsre_test_par = get_rmse(y_test_par, y_hat_test_par)
    y_data_hat_par = reg.predict(x_data_par)

    # Predicted retention times
    tg_total = model(reg, iso_data, t_void, grad_data, sc_par).flatten(order='F')

    rmsre_grad_par = get_rmse(tg_exp, tg_total)

    print('Iteration #{}/{} completed'.format(arg_iter[0] + 1, max_iter))
    return rmsre_train_par, rmsre_test_par, y_data, y_data_hat_par, rmsre_grad_par, tg_exp, tg_total


if __name__ == '__main__':
    # Display initialization and initialize start time
    print('Initiating with {} iterations'.format(max_iter))
    start_time = time()

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

    # Save the distribution of isocratic retention time errors
    DataFrame(rmse_iso, columns=['rmsre_train', 'rmsre_test']).to_csv(results_dir + '2019-QSRR_IC_PartIV-{}_{}_errors_'
                                                                                    'iso_{}_iters_run_{}.csv'.
                                                                      format(datetime.now().strftime('%d_%m_%Y-%H_%M'),
                                                                             method, max_iter, count), header=True)

    # Save predicted isocratic retention times
    y_pred = DataFrame(y_pred)
    y_pred.to_csv(results_dir + '2019-QSRR_IC_PartIV-{}_{}_logk_iso_{}_iters_run_{}.csv'
                  .format(datetime.now().strftime('%d_%m_%Y-%H_%M'),
                          method, max_iter, count), header=True)

    # Save the distribution of gradient retention time errors
    DataFrame(rmse_grad, columns=['rmsre_grad']).to_csv(
        results_dir + '2019-QSRR_IC_PartIV-{}_{}_errors_grad_{}_iters_run_{}.csv'.format(
            datetime.now().strftime('%d_%m_%Y-%H_%M'), method, max_iter, count), header=True)

    # Save predicted gradient retention times
    tg_pred = DataFrame(tg_pred)
    tg_pred.to_csv(results_dir + '2019-QSRR_IC_PartIV-{}_{}_tR_grad_{}_iters_run_{}.csv'
                   .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method,
                           max_iter, count), header=True)

    # Compute and display run-time
    run_time = time() - start_time

    # Exit flag
    exit('Simulation completed successfully !\nRun-time: {}\n'.format(strftime("%H:%M:%S", gmtime(run_time))))
