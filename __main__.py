import pandas as pd
import os
import warnings
from process import traintest, validate
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import numpy as np
import time

start_time = time.time()
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

models_final_train = list()
models_final_valid = list()

max_iter = 2

for i in range(max_iter):
    # warnings.simplefilter('ignore')
    iternum = i + 1
    print('Commencing Iteration {}'.format(iternum))
    models_final_train.append(traintest(modeldata, limxc, limdelc, n_splits, max_component, text=True))
    validate(validationdata, models_final_train[i], limxc, limdelc, text=True)
    if iternum % 10 == 0:
        elapsed_time = time.time() - start_time
        print('Iteration #{} completed in {}'.format(iternum, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

mre_final = [models_final_train[i][5] for i in range(max_iter)]
plt.hist(mre_final, bins=20, density=True)
xmin, xmax = plt.xlim()
