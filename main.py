"""

QSRR model development in IC Part IV project

Required packages:
1) Numpy
2) Pandas
3) Scipy
4) scikit-learn
5) xgBoost

Usage:
main.py max_iter count proc_i method opt_prompt n_splits (optional)

"""
from os import getcwd, makedirs
from os.path import exists
from sys import argv, stdout
from time import time, strftime, gmtime
from datetime import datetime
from pandas import read_csv, DataFrame
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from src.modules.iso2grad import ig_model as ig

""" Fixed variables 

Input arguments:
1) max_iter     : number of iterations
2) count        : run count (necessary if some parallel runs finish at the same time)
3) proc_i       : number of processes
4) method       : regression method (currently implemented: xgb, gbr, pls)
5) opt_prompt   : prompt for optimization of hyper-parameters (yes, no, default: no)
6) n_splits     : number of cross-validation splits (for optimization, if opt_prompt is no, then n_splits is [])

"""

# Show usage if no arguments are passed to the file and define variables from the input arguments
max_iter, count, proc_i, method, opt_prompt = (int(argv[1]), int(argv[2]), int(argv[3]), str(argv[4]), str(argv[5])) \
    if len(argv) > 1 else exit('Usage: python main.py max_iter count proc_i method opt_prompt n_splits')

n_splits = int(argv[6]) if opt_prompt == "yes" else []

# Make sure that method is 'xgb', 'gbr', 'pls', 'rfr', or 'ada'
assert method in ['xgb', 'gbr', 'pls', 'rfr', 'ada'], \
    'Please enter either ''pls'', ''xgb'',''rfr'',''ada'', or ''gbr'' !'

# Directories
curr_dir = getcwd()
data_dir, results_dir = curr_dir + '/data/', curr_dir + '/results/' + method + '/'

# Create the results directory if it does not exist
makedirs(results_dir) if not exists(results_dir) else []

""" 
Loading data into Pandas DataFrames & Numpy arrays:
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
----------------- (Hyper-)parameter optimization -----------------
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
    from src.modules.regr import RegressionHyperParamOpt

    # Optimization of (hyper-)parameters
    RegressionHyperParamOpt = RegressionHyperParamOpt(method, x_train_opt, y_train_opt, x_test_opt, y_test_opt,
                                                      n_splits, proc_i, results_dir, count).optimize()
    reg_params = RegressionHyperParamOpt.params_final

else:

    # List of optimized parameters
    reg_params_list = [{'n_estimators': 497, 'learning_rate': 0.23, 'max_depth': 2},
                       {'n_estimators': 485, 'learning_rate': 0.23, 'max_depth': 2}, {'n_components': 3},
                       {'n_estimators': 200, 'max_depth': 5, 'min_samples_leaf': 1},
                       {'n_estimators': 439, 'learning_rate': 0.61}]  # Changing n_components to optimal: 3

    # Default parameter conditionals
    reg_params = reg_params_list[0] if method == 'xgb' else reg_params_list[1] if method == 'gbr' \
        else reg_params_list[3] if method == 'rfr' else reg_params_list[4] if method == 'ada' \
        else reg_params_list[2] if method == 'pls' else []


""" 
----------------- Resampling with replacement -----------------
"""


# Defining a function to feed to multiprocessing
def model_parallel(arg_iter):

    # Random seed
    rnd_state = (count * max_iter) + (arg_iter[0] + 1)

    # Randomly split the data into training and testing
    x_train_unscaled_par, x_test_unscaled_par, y_train_par, y_test_par = train_test_split(x_data, y_data, test_size=0.3,
                                                                                          shuffle=True,
                                                                                          random_state=rnd_state)

    # Define a scaling object
    sc_par = StandardScaler()

    # Scale the training data and save the mean & std into the object "sc"
    x_train_par = DataFrame(sc_par.fit_transform(x_train_unscaled_par), columns=x_train_unscaled_par.columns).values

    # Scale the testing data (using the training mean & std)
    x_test_par = DataFrame(sc_par.transform(x_test_unscaled_par), columns=x_test_unscaled_par.columns).values

    # Scale all of the data (using the training mean & std)
    x_data_par = DataFrame(sc_par.transform(x_data), columns=x_data.columns).values

    # Import the RegressorsQSRR class
    from src.modules.regr import RegressorsQSRR

    # Instantiate the RegressorsQSRR class with data and run the regress() method
    reg = RegressorsQSRR(method, [x_train_par, x_test_par, y_train_par, y_test_par], reg_params).regress()

    # Predict y-values using the model
    y_data_hat_par = reg.model.predict(x_data_par).ravel()

    # Run the metrics() method to calculate the values of the model metrics
    reg = reg.metrics()

    # Predicted gradient retention times
    tg_total = ig(reg, iso_data, t_void, grad_data, sc_par).flatten(order='F')

    # Gradient retention time errors
    _, _, rmse_grad_par = reg.get_errors(tg_exp, tg_total)

    # Completion of iterations
    print('Iteration #{}/{} completed'.format(arg_iter[0] + 1, max_iter))

    # Flush the output buffer // fix the logging issues
    stdout.flush()

    return reg.rmse_train, reg.rmse_test, y_data, y_data_hat_par, rmse_grad_par, tg_exp, tg_total


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
    print('Calculations completed successfully !\nRun-time: {}\n'.format(strftime("%H:%M:%S", gmtime(run_time))))
    exit('Success !')
