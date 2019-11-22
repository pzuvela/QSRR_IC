"""

Processing results of parallel runs of main.py

Input arguments:
1) method : xgb, gbr, ada, rfr, pls

Usage:
python proc_results.py method

"""

# Importing packages
from sys import argv
from os import getcwd, makedirs
from os.path import exists
from numpy import genfromtxt
from pandas import read_csv
from src.modules.support.func import merge_files
from datetime import datetime

# Input arguments
method = exit('Usage: python proc_results.py method') if not len(argv) > 1 else str(argv[1])

# Directories
curr_dir = getcwd()
data_dir, results_dir = curr_dir + '/data/', curr_dir + '/results/' + method + '/'
merged_results_dir = curr_dir + '/results_merged/' + method + '/'

# Create the results directory if it does not exist
makedirs(merged_results_dir) if not exists(merged_results_dir) else None

# IC data for QSRR
raw_data = read_csv(data_dir + '2019-QSRR_in_IC_Part_IV_data_latest.csv')

# Drop rows with logk values of 0.000 (relative errors cannot be computed for these)
raw_data.drop(raw_data[raw_data['logk'] == 0.000].index, inplace=True)

# Define logk_iso_exp and ravel it into a column/row vector
logk_iso_exp = raw_data[['logk']].values.ravel()

# Gradient retention times
tg_exp = genfromtxt(data_dir + 'tg_data.csv', delimiter=',')

# Merge isocratic QSRR model errors and add C.I.
merge_files(results_dir+'*{}_errors_iso*'.format(method), tr_exp=None).to_csv(
    merged_results_dir + '2019-QSRR_IC_PartIV-{}_{}_errors_iso_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method), header=True)

# Merge experimental isocratic logk values with predicted ones and add C.I.
merge_files(results_dir+'*{}_logk_iso*'.format(method), tr_exp=logk_iso_exp).to_csv(
    merged_results_dir + '2019-QSRR_IC_PartIV-{}_{}_logk_iso_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method), header=True)

# Merge gradient QSRR model errors and add C.I.
merge_files(results_dir+'*{}_errors_grad*'.format(method)).to_csv(
    merged_results_dir + '2019-QSRR_IC_PartIV-{}_{}_errors_grad_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method), header=True)

# Merge experimental gradient tR values with predicted ones and add C.I.
merge_files(results_dir+'*{}_tR_grad*'.format(method), tr_exp=tg_exp).to_csv(
    merged_results_dir + '2019-QSRR_IC_PartIV-{}_{}_tR_grad_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method), header=True)

# If PLS then merge R2X and R2Y
merge_files(results_dir+'*{}_iso_perc_var*'.format(method), tr_exp=None).to_csv(
    merged_results_dir + '2019-QSRR_IC_PartIV-{}_{}_perc_var_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method), header=True) if method == 'pls' else None

"""
---------- Deprecated code
# Lengths
iso_err_len = len(glob(results_dir+'*{}_errors_iso*{}_iters_run*'.format(method, max_iter)))
iso_logk_len = len(glob(results_dir+'*{}_logk_iso*{}_iters_run*'.format(method, max_iter)))
grad_err_len = len(glob(results_dir+'*{}_errors_grad_{}_iters_run*'.format(method, max_iter)))
grad_tR_len = len(glob(results_dir+'*{}_tR_grad*{}_iters_run*'.format(method, max_iter)))
---------- Deprecated code
"""
