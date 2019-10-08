import time
import pandas as pd
import os

from process import traintest, validate
from sklearn.model_selection import train_test_split
from func import updatetable, add_true_mean_std, add_mean_std
from multiprocessing import Pool


start_time = time.time()
rawdata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
# modeldata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/HeLaK.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_sp.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')

masterStats = []
max_iter = 100
max_run = 1
proc_i = 4
limxc = [1.9, 2.2, 3.75]
limdelc = 0.08
n_splits = 3
max_component = 11


def model_parallel(arg_iter):
    modeldata, validationdata = train_test_split(rawdata, test_size=0.3, shuffle=True)
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
    stats_proba1 = updatetable(rawdata, limxc, limdelc, models, arg_iter)[0]
    stats_yhat = updatetable(rawdata, limxc, limdelc, models, arg_iter)[1]
    stats_proba2 = updatetable(rawdata, limxc, limdelc, models, arg_iter)[2]

    print('Iteration #{}/{} completed'.format(arg_iter[0] + 1, max_iter))

    return metrics, stats_proba1, stats_yhat, stats_proba2


# Execute parallelized optimization only if the file is ran as a main file
if __name__ == '__main__':

    for run_num in range(1, max_run + 1):
        print('Initiating with {} iterations, run#{}'.format(max_iter, run_num))
        run_start_time = time.time()

        # Start parallel pool for multiprocessing (processes=number of threads)
        p = Pool(processes=proc_i)

        models_final = p.map(model_parallel, zip(range(max_iter)))

        masterStats = [models_final[i][0] for i in range(max_iter)]

        y_true1 = models_final[0][1][0]
        y_proba1 = [models_final[i][1][1] for i in range(max_iter)]

        y_true2 = models_final[0][2][0]
        y_hat_reg = [models_final[i][2][1] for i in range(max_iter)]

        y_true3 = [models_final[i][3][0] for i in range(max_iter)]  # label may change for this model
        y_proba2 = [models_final[i][3][1] for i in range(max_iter)]

        run_time = time.time() - run_start_time
        print('Simulation Completed')
        print('Run Duration: {}\n'.format(time.strftime("%H:%M:%S", time.gmtime(run_time))))

        # Generating csv of metrics
        column = ['acc_train_sequest', 'sens_train_sequest', 'spec_train_sequest', 'mcc_train_sequest',
                  'rmse_train_qsrr',
                  'acc_train_both', 'sens_train_both', 'spec_train_both', 'mcc_train_both',
                  'acc_test_sequest', 'sens_test_sequest', 'spec_test_sequest', 'mcc_test_sequest',
                  'rmse_test_qsrr',
                  'acc_test_both', 'sens_test_both', 'spec_test_both', 'mcc_test_both',
                  'acc_valid_sequest', 'sens_valid_sequest', 'spec_valid_sequest', 'mcc_valid_sequest',
                  'rmse_valid_qsrr',
                  'acc_valid_both', 'sens_valid_both', 'spec_valid_both', 'mcc_valid_both']
        df_masterstats = pd.DataFrame(masterStats, columns=column)
        add_true_mean_std(None, df_masterstats).to_csv('results/iteration_metrics_{}iters_run{}.csv'
                                                       .format(max_iter, run_num), header=True)

        # Generating predictions
        df_yproba1 = pd.DataFrame(y_proba1)
        add_true_mean_std(y_true1, df_yproba1).to_csv('results/sequest_predictedprobability1_{}iters_run{}.csv'
                                                      .format(max_iter, run_num), header=True)

        df_y_hat_reg = pd.DataFrame(y_hat_reg)
        add_true_mean_std(y_true2, df_y_hat_reg).to_csv('results/qsrr_trprediction_{}iters_run{}.csv'
                                                        .format(max_iter, run_num), header=True)

        df_yproba2 = pd.DataFrame(y_proba2)
        add_true_mean_std(y_true1, df_yproba2).to_csv('results/both_predictedprobability2_{}iters_run{}.csv'
                                                      .format(max_iter, run_num), header=True)
    total_time = time.time() - start_time
    print('\nTotal Duration: {}'.format(time.strftime("%H:%M:%S", time.gmtime(total_time))))
