import pandas as pd
import os
import warnings
from process import traintest, validate
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from func import histplot, updatetable
import time

start_time = time.time()
trainingdata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
# modeldata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')

# validationdata = pd.read_csv(os.getcwd() + '/data/HeLaK.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_sp.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')

limxc = [1.9, 2.2, 3.75]
limdelc = 0.08
n_splits = 3
max_component = 11

max_iter = 1

# Pre loading variables
masterTestStats = []
masterValidStats = []
stats_df = pd.DataFrame()

print('Initiating Simulation with {} iterations'.format(max_iter))
for i in range(max_iter):
    modeldata, validationdata = train_test_split(trainingdata, test_size=0.3, shuffle=True)
    modeldata = modeldata.reset_index()
    validationdata = validationdata.reset_index()

    # warnings.simplefilter('ignore')
    models, stats = traintest(modeldata, limxc, limdelc, n_splits, max_component, text=False)
    # stats is in the following format: acc, sens, spec, mcc,    for sequest
    #                                   rmsre_train, rmsre_test, for qsrr
    #                                   acc, sens, spec, mcc     for sequest + qsrr
    masterTestStats.append(stats)

    validstats = validate(validationdata, models, limxc, limdelc, text=False)
    # validstats is in the following format: acc, sens, spec, mcc,    for sequest
    #                                        rmsre_test,              for qsrr
    #                                        acc, sens, spec, mcc     for sequest + qsrr
    masterValidStats.append(validstats)

    updatetable(stats_df, trainingdata, limxc, limdelc, models, i)

    iternum = i + 1
    if iternum % 10 == 0:
        elapsed_time = time.time() - start_time
        print('Iteration #{}/{} completed in {}'.format(iternum, max_iter,
                                                        time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
total_time = time.time() - start_time
print('Simulation Completed')
print('Total Duration: {}'.format(time.strftime("%H:%M:%S", time.gmtime(total_time))))

# sort out the metrics
# test_acc1 = [masterTestStats[i][0] for i in range(max_iter)]
# test_sens1 = [masterTestStats[i][1] for i in range(max_iter)]
# test_spec1 = [masterTestStats[i][2] for i in range(max_iter)]
test_mcc1 = [masterTestStats[i][3] for i in range(max_iter)]
# test_rmsre_train = [masterTestStats[i][4] for i in range(max_iter)]
test_rmsre_test = [masterTestStats[i][5] for i in range(max_iter)]
# test_acc2 = [masterTestStats[i][6] for i in range(max_iter)]
# test_sens2 = [masterTestStats[i][7] for i in range(max_iter)]
# test_spec2 = [masterTestStats[i][8] for i in range(max_iter)]
test_mcc2 = [masterTestStats[i][9] for i in range(max_iter)]

# valid_acc1 = [masterValidStats[i][0] for i in range(max_iter)]
# valid_sens1 = [masterValidStats[i][1] for i in range(max_iter)]
# valid_spec1 = [masterValidStats[i][2] for i in range(max_iter)]
valid_mcc1 = [masterValidStats[i][3] for i in range(max_iter)]
valid_rmsre = [masterValidStats[i][4] for i in range(max_iter)]
# valid_acc2 = [masterValidStats[i][5] for i in range(max_iter)]
# valid_sens2 = [masterValidStats[i][6] for i in range(max_iter)]
# valid_spec2 = [masterValidStats[i][7] for i in range(max_iter)]
valid_mcc2 = [masterValidStats[i][8] for i in range(max_iter)]


histplot(test_mcc1, 'MCC of Sequest', "Matthew's CC")
histplot(test_rmsre_test, 'Testing RMSRE', 'RMSRE / %')
histplot(test_mcc2, 'MCC of Sequest + QSRR', "Matthew's CC")

histplot(valid_mcc1, 'Valid MCC of Sequest', "Matthew's CC")
histplot(valid_rmsre, 'Valid Testing RMSRE', 'RMSRE / %')
histplot(valid_mcc2, 'Valid MCC of Sequest + QSRR', "Matthew's CC")
plt.show()
