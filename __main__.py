import pandas as pd
import os
import warnings
from process import traintest, validate
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
from joblib import Parallel, delayed

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

# models_final_train = list()
models_final_valid = list()

max_iter = 1000


# Function to run the training in parallel
def models_train_parallel_fun(arg_iter):
    return traintest(modeldata, limxc, limdelc, n_splits, max_component)


# Run in parallel
models_final_train = Parallel(n_jobs=1, verbose=1, backend="threading")(map(delayed(models_train_parallel_fun),
                                                                        zip(range(max_iter))))


"""
for i in range(max_iter):
    # warnings.simplefilter('ignore')
    print('## Iteration #: \n', i)
    models_final_train.append(traintest(modeldata, limxc, limdelc, n_splits, max_component))
    # validate(validationdata, models_final_train[i], limxc, limdelc)

"""

mre_final = [models_final_train[i][5] for i in range(max_iter)]
plt.hist(mre_final, bins=20, density=True)
xmin, xmax = plt.xlim()

# np.mean(mre_final), np.std(mre_final)

mean_mre, std_mre = norm.fit(mre_final)
p = norm.pdf(np.linspace(xmin, xmax, 100), mean_mre, std_mre)
plt.plot(np.linspace(xmin, xmax, 100), p)
plt.show()
