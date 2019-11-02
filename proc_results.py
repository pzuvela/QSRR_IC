"""

Processing results of parallel runs of main.py

Input arguments:
1) method   : xgb, gbr
2) max_iter : number of iterations

Usage:
python proc_results.py method max_iter

"""

# Importing packages
from sys import argv
from os import getcwd
from numpy import genfromtxt
from pandas import read_csv
from func import merge_files
from datetime import datetime
from glob import glob

# Arguments
method = str(argv[1])
max_iter = int(argv[2])

# Directories
curr_dir = getcwd()
data_dir = curr_dir + '/data/'
results_dir = curr_dir + '/results/'

# IC data for QSRR
raw_data = read_csv(data_dir + '2019-QSRR_in_IC_Part_IV_data_latest.csv')

# Drop rows with logk values of 0.000 (relative errors cannot be computed for these)
raw_data.drop(raw_data[raw_data['logk'] == 0.000].index, inplace=True)

# Define logk_iso_exp and ravel it into a column/row vector
logk_iso_exp = raw_data[['logk']].values.ravel()

# Gradient retention times
tg_exp = genfromtxt(data_dir + 'tg_data.csv', delimiter=',')

# Lengths
iso_err_len = len(glob(results_dir+'*{}_errors_iso*{}_iters_run*'.format(method, max_iter)))
iso_logk_len = len(glob(results_dir+'*{}_logk_iso*{}_iters_run*'.format(method, max_iter)))
grad_err_len = len(glob(results_dir+'*{}_errors_grad_{}_iters_run*'.format(method, max_iter)))
grad_tR_len = len(glob(results_dir+'*{}_tR_grad*{}_iters_run*'.format(method, max_iter)))

# Merge isocratic QSRR model errors and add C.I.
merge_files(results_dir+'*{}_errors_iso*{}_iters_run*'.format(method, max_iter), tr_exp=None).to_csv(
    results_dir + '2019-QSRR_IC_PartIV-{}_{}_errors_iso_{}_iters_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method, max_iter * iso_err_len), header=True)

# Merge experimental isocratic logk values with predicted ones and add C.I.
merge_files(results_dir+'*{}_logk_iso*{}_iters_run*'.format(method, max_iter), tr_exp=logk_iso_exp).to_csv(
    results_dir + '2019-QSRR_IC_PartIV-{}_{}_logk_iso_{}_iters_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method, max_iter * iso_logk_len), header=True)

# Merge gradient QSRR model errors and add C.I.
merge_files(results_dir+'*{}_errors_grad_{}_iters_run*'.format(method, max_iter)).to_csv(
    results_dir + '2019-QSRR_IC_PartIV-{}_{}_errors_grad_{}_iters_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method, max_iter * grad_err_len), header=True)

# Merge experimental gradient tR values with predicted ones and add C.I.
merge_files(results_dir+'*{}_tR_grad*{}_iters_run*'.format(method, max_iter), tr_exp=tg_exp).to_csv(
    results_dir + '2019-QSRR_IC_PartIV-{}_{}_tR_grad_{}_iters_merged.csv'
    .format(datetime.now().strftime('%d_%m_%Y-%H_%M'), method, max_iter * grad_tR_len), header=True)
