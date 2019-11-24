# ...

from os import getcwd, makedirs
from os.path import exists
from src.modules.modelling.regr import *
from numpy import log
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from csv import writer

# Method
method = 'rfr'

# Directories
curr_dir = getcwd()
data_dir, results_dir = curr_dir + '/data/', curr_dir + '/results/dyes/'

# Create the results directory if it does not exist
makedirs(results_dir) if not exists(results_dir) else []

# Load dataset
raw_data = read_csv(data_dir + 'dyes_dataset-2019-11-23.csv')

# X-data
x_data = raw_data.drop(['Name', 'logk', 'tR'], axis=1)
y_data = log(raw_data['tR'].values.ravel())

# List of optimized parameters (updated with new optimized values for ADA & RFR)
reg_params_list = [{'n_estimators': 497, 'learning_rate': 0.23, 'max_depth': 2},
                   {'n_estimators': 485, 'learning_rate': 0.23, 'max_depth': 2}, {'n_components': 2},
                   {'n_estimators': 150, 'max_depth': 15, 'min_samples_leaf': 1},
                   {'n_estimators': 676, 'learning_rate': 0.1284015, 'loss': 'exponential'}]  # Testicle

# Default parameter conditionals
reg_params = reg_params_list[0] if method == 'xgb' else reg_params_list[1] if method == 'gbr' \
    else reg_params_list[3] if method == 'rfr' else reg_params_list[4] if method == 'ada' \
    else reg_params_list[2] if method == 'pls' else []

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

# Instantiate the RegressorsQSRR class with data and run the regress() method
reg = RegressorsQSRR(method, [x_train_par, x_test_par, y_train_par, y_test_par], reg_params).regress()

# Save results
results = [y_train_par, reg.y_hat_train, y_test_par, reg.y_hat_test]

with open("out.csv", "w", newline="") as f:
    writer = writer(f)
    writer.writerows(results)
