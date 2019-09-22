import numpy as np
import random as rand
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import Bounds, minimize
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from petar.ad import ad


# input from preprocessing
# splitting(df)[1] should be x_train ...


def regress_pls(traintestset, n_splits, imax, optimise=False, text=False):
    x_train, x_test, y_train, y_test = traintestset
    x_train = x_train.values

    # Partial Least Squares (PLS) Method

    if optimise is True:
        # Creating initial values
        print(' ')
        print('    --- Commencing Optimisation of Regression Model ---')
        nlvs = np.zeros(imax - 1)
        # rmsee = np.zeros(imax - 1)
        rmsecv = np.zeros(imax - 1)
        for i in range(1, imax):
            # Initiate cross validation (CV) model
            pls_cv = PLSRegression(n_components=i)

            # K-Fold Object
            kf = KFold(n_splits=n_splits)

            # Pre-loading variables
            rmse_train = np.zeros(n_splits)
            rmse_test = np.zeros(n_splits)
            j = 0

            # # Construct Scoring Object
            # def rmse(y_true, y_pred):
            #     return np.sqrt(np.square(100 * (y_pred - y_true) / y_true).mean())
            # scorer = make_scorer(rmse, greater_is_better=False)
            #
            # rmse_test = cross_val_score(pls_cv, x_train, y_train, cv=kf, scoring=scorer)

            for train_i, test_i in kf.split(x_train, y_train):
                # Defining the training set
                x_train_cv = x_train[train_i]
                y_train_cv = y_train[train_i]

                # Defining the left out set
                x_test_cv = x_train[test_i]
                y_test_cv = y_train[test_i]

                # Build PLS Model
                pls_cv.fit(x_train_cv, y_train_cv)

                # Generating RMSE
                y_test_hat_cv = pls_cv.predict(x_test_cv).ravel()
                rmse_test[j] = (np.sqrt(mean_squared_error(y_test_cv, y_test_hat_cv)))
                # rmse_test[j] = 100 * np.sqrt(np.mean(np.square((y_test_hat_cv - y_test_cv) / y_test_cv)))

                # y_train_hat = pls_cv.predict(x_train).ravel()
                # rmse_train[j] = np.sqrt(mean_squared_error(y_train, y_train_hat))
                # rmse_train[j] = np.sqrt(np.mean(np.square(np.subtract(y_train, y_train_hat))))
                j += 1

            # Gathering Statistic
            nlvs[i - 1] = i
            # rmsee[i - 1] = np.mean(rmse_train)
            rmsecv[i - 1] = np.mean(rmse_test)

        # Implementing optimised parameters
        # optstats = [nlvs, rmsecv, rmsee]
        optstats = [nlvs, rmsecv]
        lvs = knee(nlvs, rmsecv)  # some code to find knee point
        print('    Optimised Lvs: {}'.format(lvs))
        print('    ------------ Completion of Optimisation -----------')
        print(' ')
    else:
        optstats = None
        lvs = 5

    if text is True:
        print('The number of nLVs used is {}'.format(lvs))
    pls = PLSRegression(n_components=lvs)
    pls.fit(x_train, y_train)
    y_hat_test = pls.predict(x_test)
    y_hat_train = pls.predict(x_train)
    x_train = pd.DataFrame(x_train, columns=x_test.columns)

    return pls, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test], optstats


def regress_gbr(traintestset, n_splits, optimise=False):
    x_train, x_test, y_train, y_test = traintestset

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()

    if optimise is True:
        print(' ')
        print('    --- Commencing Optimisation of Regression Model ---')

        def fun(x):
            # Descaling Parameters
            n_est = int(np.round(np.exp(x[0]), decimals=0))
            min_sam = int(np.round(np.exp(x[1]), decimals=0))
            lr = x[2] ** 2
            max_depth = int(np.round(np.exp(x[3]), decimals=0))
            print(n_est, min_sam, lr, max_depth)
            opt_gbr = GradientBoostingRegressor(n_estimators=n_est,
                                                min_samples_split=min_sam,
                                                learning_rate=lr,
                                                max_depth=max_depth)

            # K-Fold object
            kfold = KFold(n_splits=n_splits)

            # Scoring object
            def rmse(y_true, y_pred):
                return np.sqrt(np.square(100 * (y_pred - y_true) / y_true).mean())

            scorer = make_scorer(rmse, greater_is_better=False)
            # CV score
            score = cross_val_score(opt_gbr, x_train, y_train, cv=kfold, scoring=scorer)
            print(np.mean(score))

            return np.mean(score)

        def constr_fun(x):
            c = (np.exp(x[0]) / x[2]**2) - x[4]
            return c

        # Creating bounds
        n_est_min, n_est_max = 50, 500
        min_sam_min, min_sam_max = 2, 10
        lr_min, lr_max = 0.01, 0.99
        max_depth_min, max_depth_max = 1, 5
        bounds = Bounds([np.log(n_est_min), np.log(min_sam_min), np.sqrt(lr_min), np.log(max_depth_min),
                         n_est_min / lr_min],
                        [np.log(n_est_max), np.log(min_sam_max), np.sqrt(lr_max), np.log(max_depth_max),
                         n_est_max / lr_max])

        # Pre-loading initial values
        n_est0 = np.log(rand.uniform(n_est_min, n_est_max))
        min_sam0 = np.log(rand.uniform(min_sam_min, min_sam_max))
        lr0 = np.sqrt(rand.uniform(lr_min, lr_max))
        max_depth0 = np.log(rand.uniform(max_depth_min, max_depth_max))
        k0 = rand.uniform(n_est_min / lr_min, n_est_max / lr_max)
        initial = np.array([n_est0, min_sam0, lr0, max_depth0, k0])
        print('    ---------------- Initial Parameters ---------------')
        print('    n_estimators: {:.0f}\n'
              '    min_sample_split: {:.0f}\n'
              '    learning_rate: {:.2f} \n'
              '    max_depth: {:.0f}'
              .format(np.exp(n_est0), np.exp(min_sam0), np.square(lr0), np.exp(max_depth0))
              )
        print('    ---------------------------------------------------')

        # Begin Optimisation
        opt = minimize(fun, initial, method='trust-constr', constraints={'fun': constr_fun, 'type': 'eq'},
                       bounds=bounds, options={'maxiter': 100, 'disp': True})
        x_dict = {'n_estimators': int(np.round(np.exp(opt.x[0]), decimals=0)),
                  'min_samples_split': int(np.round(np.exp(opt.x[1]), decimals=0)),
                  'learning_rate': np.square(opt.x[2]),
                  'max_depth': int(np.round(np.exp(opt.x[3]), decimals=0))
                  }

        # Implementing optimised parameters
        gbr.set_params(**x_dict)

        print('    ----------------- Final Parameters ----------------')
        print('    n_estimators: {:.0f}\n'
              '    min_sample_split: {:.0f}\n'
              '    learning_rate: {:.2f} \n'
              '    max_depth: {:.0f}\n'
              '    Final CV-RMSE: {:.2f}'
              .format(np.exp(opt.x[0]), np.exp(opt.x[1]), np.square(opt.x[2]), np.exp(opt.x[3]), opt.fun)
              )
        print('    ---------------------------------------------------')
        print('    ------------ Completion of Optimisation -----------')
        print(' ')

    gbr.fit(x_train, y_train)
    y_hat_train = gbr.predict(x_train)
    y_hat_test = gbr.predict(x_test)

    return gbr, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test], None


def regress_stats(traindata, testdata, text=False):
    if traindata is not None:
        x_train, y_train, y_hat_train = traindata
        msre_train = np.square(100 * (y_hat_train - y_train) / y_train).mean()
        rmsre_train = np.sqrt(msre_train)

        x_test, y_test, y_hat_test = testdata
        msre_test = np.square(100 * (y_hat_test - y_test) / y_test).mean()
        rmsre_test = np.sqrt(msre_test)
        # Printing metrics
        if text is True:
            print('------------- Regression Model Stats --------------')
            print('The training RMSRE is {:.2f}%'.format(rmsre_train))
            print('The testing RMSRE is {:.2f}%'.format(rmsre_test))
            print('--------------- End of Statistics -----------------')
        return rmsre_train, rmsre_test
    else:
        x_test, y_test, y_hat_test = testdata
        msre_test = np.square(100 * (y_hat_test - y_test) / y_test).mean()
        rmsre_test = np.sqrt(msre_test)

        # Printing metrics
        if text is True:
            print('------------- Regression Model Stats --------------')
            print('The testing RMSRE is {:.2f}%'.format(rmsre_test))
            print('--------------- End of Statistics -----------------')
        return rmsre_test


def regress_plot(teststat, trainstat=None, optstat=None):
    x_test, y_test, y_hat_test = teststat
    y_hat_test = y_hat_test.ravel()

    test_residue = y_hat_test - y_test

    # res histogram
    fig7, ax7 = plt.subplots()
    ax7.hist(test_residue, color='C0', density=True)

    # residue plot
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, test_residue, c='C0', label='Test Set')
    lim1 = [np.min(ax1.get_xlim()), np.max(ax1.get_xlim())]
    lim2 = [0, 0]
    ax1.plot(lim1, lim2, c='k')
    ax1.set_xlim(lim1)
    ax1.set_xlabel('Retention Time')
    ax1.set_ylabel('Residue')
    ax1.set_title('Residue of prediction')
    ax1.legend()

    # response plot
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_hat_test, c='C0', label='Test Set')
    lims = [
        np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
        np.max([ax2.get_xlim(), ax2.get_ylim()])  # max of both axes
    ]
    ax2.plot(lims, lims, c='k')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Response Plot')
    ax2.legend()

    if trainstat is not None:
        x_train, y_train, y_hat_train = trainstat
        y_hat_train = y_hat_train.ravel()

        train_residue = y_hat_train - y_train

        ad(y_train, y_hat_train, y_test, y_hat_test, x_train, x_test, 'yes')

        ax7.hist(train_residue, color='C1', alpha=0.7, density=True)

        ax1.scatter(y_train, train_residue, alpha=0.7, c='C1', label='Train Set')
        ax1.legend()

        ax2.scatter(y_train, y_hat_train, alpha=0.7, c='C1', label='Train Set')
        ax2.legend()

    if optstat is not None:
        [x_values, rmsecv] = optstat

        # LV optimisation plot
        fig3, ax3 = plt.subplots()
        ax3.plot(x_values, rmsecv, label='RMSECV')
        # ax3.plot(x_values, rmsee, label='RMSEE')
        ax3.set_xticks(x_values)
        ax3.set_xlabel('Number of LVs')
        ax3.set_ylabel('Error')
        ax3.set_title('Optimisation of LVs')
        ax3.legend()


def add_error(reg, data, scaleddata, traindata=None, text=False):
    # addition of error into data
    x_data_reg = scaleddata[0][['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = scaleddata[1]['tR / min'].values
    y_hat_all = reg.predict(x_data_reg)

    re = 100 * np.abs((y_hat_all.ravel() - y_data) / y_data)

    data.loc[:, 'error'] = re.ravel()

    if traindata is not None:
        # calculation of mre
        x_train, y_train, y_hat_train = traindata
        msre = np.square(100 * (y_hat_train - y_train) / y_train).mean()
        rmsre = np.sqrt(msre)
        if text is True:
            print('The training RMSRE is {:.2f}%'.format(rmsre))
        return data, rmsre
    else:
        return data


def knee(nlvs, rmsecv):
    del1, del2 = [], []
    for i in range(0, len(nlvs)-1):
        del1.append(rmsecv[i+1] - rmsecv[i])

    for j in range(0, len(del1)-1):
        del2.append(del1[j+1] - del1[j])

    max_del2 = np.argmax(del2)
    kneept = nlvs[max_del2 + 2]
    return int(kneept)

