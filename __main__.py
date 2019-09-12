import pandas as pd
import os
import warnings
from process import traintest, validate
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import numpy as np

print('Initiating Input Data and Selected parameters...', end=' ')
modeldata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
# modeldata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')

modeldata, validationdata = train_test_split(modeldata, test_size=0.3, shuffle=True)
modeldata = modeldata.reset_index()
validationdata = validationdata.reset_index()

# validationdata = pd.read_csv(os.getcwd() + '/data/HeLaK.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_sp.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')

limxc = [1.9, 2.2, 3.75]
limdelc = 0.08
n_splits = 3
max_component = 11
print('Initiated')
print('')

models_final_train = list()
models_final_valid = list()

max_iter = 1

for i in range(max_iter):
    # warnings.simplefilter('ignore')
    print('## Iteration #: ', i)
    models_final_train.append(traintest(modeldata, limxc, limdelc, n_splits, max_component))
    validate(validationdata, models_final_train[i], limxc, limdelc)

mre_final = [models_final_train[i][5] for i in range(max_iter)]
plt.hist(mre_final, bins=20, density=True)
xmin, xmax = plt.xlim()

# np.mean(mre_final), np.std(mre_final)

# mean_mre, std_mre = norm.fit(mre_final)
# p = norm.pdf(np.linspace(xmin, xmax, 100), mean_mre, std_mre)
# plt.plot(np.linspace(xmin, xmax, 100), p)
# plt.show()
