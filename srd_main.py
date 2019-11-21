# Import modules
from os import getcwd
from sys import argv
from glob import glob
from datetime import datetime
from numpy import genfromtxt, ndarray
from pandas import DataFrame
from src.modules.srd import SumOfRankingDiffs
from src.modules.func import add_true_mean_std

# Input arguments
method, y_var = exit('Usage: python srd_main.py method y_var n_rnd_vals') if not len(argv) > 1 else \
                    [str(argv[i]) for i in range(1, 3, 1)]
n_rnd_vals = [] if len(argv) < 4 else argv[3]

# Assert whether method is 'xgb', 'gbr', 'pls', 'rfr', or 'ada'
assert method in ['xgb', 'gbr', 'pls', 'rfr', 'ada'], \
    '# Please enter either \'xgb\', \'gbr'', \'ada\', \'rfr\', or \'pls\' !'

# Assert whether y_var is in the list of parameters
assert y_var in ['logk_iso', 'tR_grad'], '# Please enter either \'logk_iso\', or \'tR_grad\' !'

# Define an empty ndarray
res = ndarray([])

# Open the csv results file
for file in glob(getcwd() + '/results_merged/'+method+'/*'+method+'*'+y_var+'*.csv'):
    print('# Loading results file: {}. Please wait ...'.format(file))
    res = genfromtxt(file, delimiter=',', skip_header=1)

# Targets
t, x = res[0, 1:].T, res[6:, 1:].T

# Instantiate the SumOfRankingDiffs object
srd = SumOfRankingDiffs(a=x, t=t)

# Compute SRDs
print('# Computing SRDs. Please wait...')
srd.compute_srd()

# Normalize SRDs
srd.srd_normalize()

# Validate SRDs
print('# Validating SRDs with {} random numbers. Please wait...'.format(10000 if not n_rnd_vals else n_rnd_vals))
srd.srd_validate() if not n_rnd_vals else srd.srd_validate(n_rnd_vals=n_rnd_vals)

# Export files
add_true_mean_std(None, DataFrame(srd.srd_norm)).to_csv(
    getcwd() + '/results_merged/srd/2019-QSRR_IC_PartIV-{}_{}_{}-SRD_norm.csv'.format(
        datetime.now().strftime('%d_%m_%Y-%H_%M'), method, y_var))
add_true_mean_std(None, DataFrame(srd.srd_rnd_norm)).to_csv(
    getcwd() + '/results_merged/srd/2019-QSRR_IC_PartIV-{}_{}_{}-SRD_rnd_norm.csv'.format(
        datetime.now().strftime('%d_%m_%Y-%H_%M'), method, y_var))

