import pandas as pd
import os
import warnings
from process import traintest, validate

print('Initiating Input Data and Selected parameters...', end=' ')
modeldata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
# modeldata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/HeLaK.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_sp.csv')
validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')
limxc = [1.9, 2.2, 3.75]
limdelc = 0.08
n_splits = 3
max_component = 11
print('Initiated')
print('')

warnings.simplefilter('ignore')
models = traintest(modeldata, limxc, limdelc, n_splits, max_component)

validate(validationdata, models, limxc, limdelc)
