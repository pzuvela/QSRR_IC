import time
import numpy as np
import pandas as pd
import os
import func

from process import traintest, validate
from scipy import optimize as opt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor, XGBClassifier
from multiprocessing import Pool
from preprocessing import labelling, splitting


start_time = time.time()
rawdata = pd.read_csv(os.getcwd() + '/data/model_protein.csv')
# modeldata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/HeLaK.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_sp.csv')
# validationdata = pd.read_csv(os.getcwd() + '/data/Bacillus_subtilis_deltaPrpE.csv')

max_iter = 100
proc_i = 4
limxc = [1.9, 2.2, 3.75]
limdelc = 0.08
n_splits = 5

# Optimisation procedure
if input('Initiate Optimisation? (Y/N)').lower() == 'y':
    df = labelling(rawdata, limxc, limdelc, None, 'delc')
    scaled_data, labelset, trset, sc, tr_max = splitting(df, 'Sequest')

    x_train, x_test, y_train, y_test = labelset
    clf = XGBClassifier().fit(x_train, y_train)
    initial = clf.get_params()

    y_hat = clf.predict(x_train)
    initial_rmsre_train = func.get_rmsre(y_train, y_hat)

    y_hat = clf.predict(x_test)
    initial_rmsre_test = func.get_rmsre(y_test, y_hat)

    opt_start = time.time()

    print('    ---------------- Initial Parameters ---------------')
    print('    n_estimators: {:.0f}\n'
          '    learning_rate: {:.2f} \n'
          '    max_depth: {:.0f}\n'
          '    Initial %RMSEE: {:.2f}'
          '    Initial %RMSEP: {:.2f}'
          .format(initial['n_estimators'],
                  initial['learning_rate'],
                  initial['max_depth'],
                  initial_rmsre_train,
                  initial_rmsre_test
                  )
          )
    print('    ---------------------------------------------------')

    # Creating optimisation function, needs to be in each
    def objective(x):
        # Descaling Parameters
        n_est = int(np.round(x[0], decimals=0))
        lr = x[1]
        max_depth = int(np.round(x[2], decimals=0))

        opt_model = clf.set_params(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)

        # CV score
        scorer = make_scorer(func.get_rmsre)
        score = cross_val_score(opt_model, x_train, y_train, cv=KFold(n_splits=n_splits), scoring=scorer)

        return np.mean(score)


    # Creating bounds
    n_est_min, n_est_max = 10, 500
    lr_min, lr_max = 0.01, 0.9
    max_depth_min, max_depth_max = 1, 5
    bounds = opt.Bounds([n_est_min, lr_min, max_depth_min],
                        [n_est_max, lr_max, max_depth_max])

    final_values = opt.differential_evolution(objective, bounds, workers=-1, updating='deferred',
                                              mutation=(1.5, 1.9), popsize=20)
    reg_params = {'n_estimators': int(np.round(final_values.x[0], decimals=0)),
                  'learning_rate': final_values.x[1],
                  'max_depth': int(np.round(final_values.x[2], decimals=0))
                  }

    clf.set_params(**reg_params).fit(x_train, y_train)

    y_hat = clf.predict(x_train)
    final_rmsre_train = func.get_rmsre(y_train, y_hat)

    y_hat = clf.predict(x_test)
    final_rmsre_test = func.get_rmsre(y_test, y_hat)

    elapsed_time = time.time() - opt_start

    print('    ----------------- Final Parameters ----------------')
    print('    n_estimators: {:.0f}\n'
          '    learning_rate: {:.2f} \n'
          '    max_depth: {:.0f}\n'
          '    Final %RMSECV: {:.3f}\n'
          '    Final %RMSEE: {:.2f}'
          '    Final %RMSEP: {:.2f}\n'
          '    Optimisation Duration: {}'
          .format(int(np.round(final_values.x[0], decimals=0)),
                  final_values.x[1],
                  int(np.round(final_values.x[2], decimals=0)),
                  final_values.objective,
                  final_rmsre_train,
                  final_rmsre_test,
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                  )
          )
    print('    ---------------------------------------------------')

    x_train, x_test, y_train, y_test = trset
    reg = XGBRegressor(objective="reg:squarederror").fit(x_train, y_train)
    initial = reg.get_params()

    y_hat = reg.predict(x_train)
    initial_rmsre_train = func.get_rmsre(y_train, y_hat)

    y_hat = reg.predict(x_test)
    initial_rmsre_test = func.get_rmsre(y_test, y_hat)

    opt_start = time.time()

    print('    ---------------- Initial Parameters ---------------')
    print('    n_estimators: {:.0f}\n'
          '    learning_rate: {:.2f} \n'
          '    max_depth: {:.0f}\n'
          '    Initial %RMSEE: {:.2f}'
          '    Initial %RMSEP: {:.2f}'
          .format(initial['n_estimators'],
                  initial['learning_rate'],
                  initial['max_depth'],
                  initial_rmsre_train,
                  initial_rmsre_test
                  )
          )
    print('    ---------------------------------------------------')

    # Creating optimisation function, needs to be in each
    def objective(x):
        # Descaling Parameters
        n_est = int(np.round(x[0], decimals=0))
        lr = x[1]
        max_depth = int(np.round(x[2], decimals=0))

        opt_model = reg.set_params(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)

        # CV score
        scorer = make_scorer(func.get_rmsre)
        score = cross_val_score(opt_model, x_train, y_train, cv=KFold(n_splits=n_splits), scoring=scorer)

        return np.mean(score)


    # Creating bounds
    n_est_min, n_est_max = 10, 500
    lr_min, lr_max = 0.01, 0.9
    max_depth_min, max_depth_max = 1, 5
    bounds = opt.Bounds([n_est_min, lr_min, max_depth_min],
                        [n_est_max, lr_max, max_depth_max])

    final_values = opt.differential_evolution(objective, bounds, workers=-1, updating='deferred',
                                              mutation=(1.5, 1.9), popsize=20)
    reg_params = {'n_estimators': int(np.round(final_values.x[0], decimals=0)),
                  'learning_rate': final_values.x[1],
                  'max_depth': int(np.round(final_values.x[2], decimals=0))
                  }

    reg.set_params(**reg_params).fit(x_train, y_train)

    y_hat = reg.predict(x_train)
    final_rmsre_train = func.get_rmsre(y_train, y_hat)

    y_hat = reg.predict(x_test)
    final_rmsre_test = func.get_rmsre(y_test, y_hat)

    elapsed_time = time.time() - opt_start

    print('    ----------------- Final Parameters ----------------')
    print('    n_estimators: {:.0f}\n'
          '    learning_rate: {:.2f} \n'
          '    max_depth: {:.0f}\n'
          '    Final %RMSECV: {:.3f}\n'
          '    Final %RMSEE: {:.2f}'
          '    Final %RMSEP: {:.2f}\n'
          '    Optimisation Duration: {}'
          .format(int(np.round(final_values.x[0], decimals=0)),
                  final_values.x[1],
                  int(np.round(final_values.x[2], decimals=0)),
                  final_values.objective,
                  final_rmsre_train,
                  final_rmsre_test,
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                  )
          )
    print('    ---------------------------------------------------')
else:
    clf_params = None
    reg_params = None


def parallel_model(arg_iter):
    modeldata, validationdata = train_test_split(rawdata, test_size=0.3, shuffle=True, random_state=2)
    modeldata = modeldata.reset_index()
    validationdata = validationdata.reset_index()

    models, stats = traintest(modeldata, limxc, limdelc, clf_params, reg_params)
    # stats is in the following format: acc, sens, spec, mcc,    for sequest
    #                                   %rmse_train, %rmse_test, for qsrr
    #                                   acc, sens, spec, mcc     for sequest + qsrr

    validstats = validate(validationdata, models, limxc, limdelc)
    # validstats is in the following format: acc, sens, spec, mcc,    for sequest
    #                                        %rmse_test,              for qsrr
    #                                        acc, sens, spec, mcc     for sequest + qsrr

    # Collation of Statistics
    metrics = stats + validstats
    all_stats = func.get_stats(rawdata, limxc, limdelc, models)
    stats_proba1 = all_stats[0]
    stats_yhat = all_stats[1]
    stats_proba2 = all_stats[2]

    return metrics, stats_proba1, stats_yhat, stats_proba2


# Execute parallelized optimization only if the file is ran as a main file
if __name__ == '__main__':
    print('Initiating resampling with replacement for {} iterations'.format(max_iter))
    start_time = time.time()

    # Start parallel pool for multiprocessing (processes=number of threads)
    p = Pool(processes=proc_i)
    final_model = p.map(parallel_model, zip(range(max_iter)))

    # Collation of Statistics
    masterStats = [final_model[i][0] for i in range(max_iter)]

    y_true_label = final_model[0][1][0]
    y_proba1 = [final_model[i][1][1] for i in range(max_iter)]

    y_true_tr = final_model[0][2][0]
    y_hat_reg = [final_model[i][2][1] for i in range(max_iter)]

    # y_true3 = [final_model[i][3][0] for i in range(max_iter)]
    y_proba2 = [final_model[i][3][1] for i in range(max_iter)]

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
    func.add_true_mean_std(None, pd.DataFrame(masterStats, columns=column)).to_csv('results/iteration_metrics_{}iters.csv'
                                                                              .format(max_iter), header=True)

    # Generating predictions
    func.add_true_mean_std(y_true_label, pd.DataFrame(y_proba1)).to_csv('results/sequest_predictedprobability1_{}iters.csv'
                                                                   .format(max_iter), header=True)
    func.add_true_mean_std(y_true_tr, pd.DataFrame(y_hat_reg)).to_csv('results/qsrr_trprediction_{}iters.csv'
                                                                 .format(max_iter), header=True)
    func.add_true_mean_std(y_true_label, pd.DataFrame(y_proba2)).to_csv('results/both_predictedprobability2_{}iters.csv'
                                                                   .format(max_iter), header=True)

    total_time = time.time() - start_time

    print('\nTotal Duration: {}'.format(time.strftime("%H:%M:%S", time.gmtime(total_time))))
