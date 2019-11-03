"""

QSRR model development in IC Part IV project

Required packages:
1) Pandas
2) Numpy
3) Scipy
4) xgBoost
5) scikit-learn

Usage:
main.py max_iter count proc_i method opt_prompt n_splits (optional)

"""
from os import getcwd
from sys import argv
from time import time, strftime, gmtime
from datetime import datetime
from pandas import read_csv, DataFrame
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from src.modules.func import get_rmse
from src.modules.iso2grad import model

# Directories
curr_dir = getcwd()
data_dir, results_dir = curr_dir + '/data/', curr_dir + '/results/'

""" Fixed variables 

Input arguments:
1) max_iter     : number of iterations
2) count        : run count (necessary if some parallel runs finish at the same time)
3) proc_i       : number of processes
4) method       : regression method (currently implemented: xgb, gbr)
5) opt_prompt   : prompt for optimization of hyper-parameters (yes, no, default: no)
6) n_splits     : number of cross-validation splits (for optimization, if opt_prompt is no, then n_splits is [])

"""

# Show usage if no arguments are passed to the file
if not len(argv) > 1:
    exit('Usage: python main.py max_iter count proc_i method opt_prompt n_splits')

# Define variables from the input arguments
max_iter, count, proc_i, method, opt_prompt = int(argv[1]), int(argv[2]), int(argv[3]), str(argv[4]), str(argv[5])
n_splits = int(argv[6]) if opt_prompt == "yes" else []


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

""" Initial data processing """
# Drop rows with logk values of 0.000 (relative errors cannot be computed for these)
raw_data.drop(raw_data[raw_data['logk'] == 0.000].index, inplace=True)

# Drop retention times and logk values sto generate the x_data matrix, define y_data and ravel it into a vector
x_data, y_data = raw_data.drop(['tR', 'logk'], axis=1), raw_data[['logk']].values.ravel()

""" 
(Hyper-)parameter optimization
"""
# Optimization conditional
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

    # Import the optimization function from the "regr" module
    from src.modules.regr import optimization

    # Optimization of (hyper-)parameters
    reg_params = optimization(method, x_train_opt, y_train_opt, x_test_opt, y_train_opt,
                              n_splits, proc_i, results_dir, count)

else:

    # List of default parameters
    reg_params_list = [{'n_estimators': 497, 'learning_rate': 0.23, 'max_depth': 2},
                       {'n_estimators': 485, 'learning_rate': 0.23, 'max_depth': 2}, {'latent_variables': 4}]

    # Default parameter conditionals
    reg_params = reg_params_list[0] if method == 'xgb' else reg_params_list[1] if method == 'gbr' else \
        reg_params_list[2] if method == 'pls' else reg_params_list[0]


""" Resampling with replacement  """


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
        from src.modules.regr import regress_xgbr
        reg, _, _ = regress_xgbr(trset, reg_params=reg_params)

    # GBR
    elif method == 'gbr':
        from src.modules.regr import regress_gbr
        reg, _, _ = regress_gbr(trset, reg_params=reg_params)

    # Default (XGB)
    else:
        from src.modules.regr import regress_xgbr
        reg, _, _ = regress_xgbr(trset, reg_params=reg_params)

    y_hat_train_par, y_hat_test_par, y_data_hat_par = reg.predict(x_train_par), reg.predict(x_test_par), \
        reg.predict(x_data_par)
    rmsre_train_par, rmsre_test_par = get_rmse(y_train_par, y_hat_train_par), get_rmse(y_test_par, y_hat_test_par)

    # Predicted retention times
    tg_total = model(reg, iso_data, t_void, grad_data, sc_par).flatten(order='F')

    rmsre_grad_par = get_rmse(tg_exp, tg_total)

    print('Iteration #{}/{} completed'.format(arg_iter[0] + 1, max_iter))
    return rmsre_train_par, rmsre_test_par, y_data, y_data_hat_par, rmsre_grad_par, tg_exp, tg_total


# Main section
if __name__ == '__main__':
    # Display initialization and initialize start time
    print('Initiating with {} iterations'.format(max_iter))
    start_time = time()

    # Start Parallel Pool with "proc_i" processes
    p = Pool(processes=proc_i)

    # Run the model_parallel function for max_iter times
    models_final = p.map(model_parallel, zip(range(max_iter)))

    # Training and testing isocratic RMSE & gradient RMSE
    rmse_iso, rmse_grad = [models_final[i][:2] for i in range(max_iter)], [models_final[i][4] for i in range(max_iter)]

    # True and predicted isocratic & gradient retention times
    y_true, y_pred, tg_true, tg_pred = models_final[0][2], [models_final[i][3] for i in range(max_iter)], \
        models_final[0][5], [models_final[i][6] for i in range(max_iter)]

    # Save the distribution of isocratic retention time errors
    DataFrame(rmse_iso, columns=['rmsre_train', 'rmsre_test']).to_csv(results_dir + '2019-QSRR_IC_PartIV-{}_{}_errors_'
                                                                                    'iso_{}_iters_run_{}.csv'.
                                                                      format(datetime.now().strftime('%d_%m_%Y-%H_%M'),
                                                                             method, max_iter, count), header=True)

    # Save predicted isocratic retention times
    DataFrame(y_pred).to_csv(results_dir + '2019-QSRR_IC_PartIV-{}_{}_logk_iso_{}_iters_run_{}.csv'
                             .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method, max_iter, count), header=True)

    # Save the distribution of gradient retention time errors
    DataFrame(rmse_grad, columns=['rmsre_grad']).to_csv(
        results_dir + '2019-QSRR_IC_PartIV-{}_{}_errors_grad_{}_iters_run_{}.csv'.format(
            datetime.now().strftime('%d_%m_%Y-%H_%M'), method, max_iter, count), header=True)

    # Save predicted gradient retention times
    DataFrame(tg_pred).to_csv(results_dir + '2019-QSRR_IC_PartIV-{}_{}_tR_grad_{}_iters_run_{}.csv'
                              .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method, max_iter, count), header=True)

    # Compute and display run-time
    run_time = time() - start_time

    # Exit flag
    exit('Simulation completed successfully !\nRun-time: {}\n'.format(strftime("%H:%M:%S", gmtime(run_time))))
