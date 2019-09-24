import pandas as pd
import os
import warnings
from process import traintest, validate
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from func import histplot, updatetable
import time
from multiprocessing import Pool


start_time = time.time()
trainingdata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
# modeldata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')

# validationdata = pd.read_csv(os.getcwd() + '/data/HeLaK.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_sp.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')

masterStats = []
dfstats_proba1 = pd.DataFrame()
dfstats_yhat = pd.DataFrame()
dfstats_proba2 = pd.DataFrame()
max_iter = 10
proc_i = 4
limxc = [1.9, 2.2, 3.75]
limdelc = 0.08
n_splits = 3
max_component = 11


def model_parallel(arg_iter):
    modeldata, validationdata = train_test_split(trainingdata, test_size=0.3, shuffle=True)
    modeldata = modeldata.reset_index()
    validationdata = validationdata.reset_index()

    # warnings.simplefilter('ignore')
    models, stats = traintest(modeldata, limxc, limdelc, n_splits, max_component, text=False)
    # stats is in the following format: acc, sens, spec, mcc,    for sequest
    #                                   rmsre_train, rmsre_test, for qsrr
    #                                   acc, sens, spec, mcc     for sequest + qsrr

    validstats = validate(validationdata, models, limxc, limdelc, text=False)
    # validstats is in the following format: acc, sens, spec, mcc,    for sequest
    #                                        rmsre_test,              for qsrr
    #                                        acc, sens, spec, mcc     for sequest + qsrr

    metrics = stats + validstats
    stats_proba1 = updatetable(trainingdata, limxc, limdelc, models, arg_iter)[0]
    stats_yhat = updatetable(trainingdata, limxc, limdelc, models, arg_iter)[1]
    stats_proba2 = updatetable(trainingdata, limxc, limdelc, models, arg_iter)[2]

    elapsed_time = time.time() - start_time
    print('Iteration #{}/{} completed in {}'.format(arg_iter, max_iter,
                                                    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    return metrics, stats_proba1, stats_yhat, stats_proba2


# Execute parallelized optimization only if the file is ran as a main file
if __name__ == '__main__':

    # Pre loading variables
    # masterTestStats = []
    # masterValidStats = []

    print('Initiating Simulation with {} iterations'.format(max_iter))

    # Start parallel pool for multiprocessing (processes=number of threads)
    p = Pool(processes=proc_i)

    models_final = p.map(model_parallel, zip(range(max_iter)))

    masterStats = [models_final[i][0] for i in range(max_iter)]
    dfstats_proba1 = pd.concat([models_final[i][1] for i in range(max_iter)], axis=1)
    dfstats_yhat = pd.concat([models_final[i][2] for i in range(max_iter)], axis=1)
    dfstats_proba2 = pd.concat([models_final[i][3] for i in range(max_iter)], axis=1)

    total_time = time.time() - start_time
    print('Simulation Completed')
    print('Total Duration: {}'.format(time.strftime("%H:%M:%S", time.gmtime(total_time))))

    # Generating csv of predictions and metrics
    column = ['acc_test_sequest', 'sens_test_sequest', 'spec_test_sequest', 'mcc_test_sequest',
              'rmse_train_qsrr', 'rmse_test_qsrr',
              'acc_test_both', 'sens_test_both', 'spec_test_both', 'mcc_test_both',
              'acc_valid_sequest', 'sens_valid_sequest', 'spec_valid_sequest', 'mcc_valid_sequest',
              'rmse_valid_qsrr',
              'acc_valid_both', 'sens_valid_both', 'spec_valid_both', 'mcc_valid_both']
    pd.DataFrame(masterStats, columns=column).to_csv('iteration_metrics.csv', header=True)
    dfstats_proba1.to_csv('sequest_predictedprobability1.csv', header=True)
    dfstats_yhat.to_csv('qsrr_trprediction.csv', header=True)
    dfstats_proba2.to_csv('both_predictedprobability2.csv', header=True)

    # plotting out selected metrics
    valid_mcc1 = [masterStats[i][13] for i in range(max_iter)]
    valid_rmsre = [masterStats[i][14] for i in range(max_iter)]
    valid_mcc2 = [masterStats[i][18] for i in range(max_iter)]

    histplot(valid_mcc1, 'Valid MCC of Sequest', "Matthew's CC")
    histplot(valid_rmsre, 'Valid Testing RMSRE', 'RMSRE / %')
    histplot(valid_mcc2, 'Valid MCC of Sequest + QSRR', "Matthew's CC")
    plt.show()
