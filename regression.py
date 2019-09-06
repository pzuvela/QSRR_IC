import numpy as np
import random as rand
from matplotlib import pyplot as plt
from scipy.optimize import Bounds, minimize
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer


# input from preprocessing
# splitting(df)[1] should be x_train ...


def regress_pls(traintestset, n_splits, imax, optimise=False):
    x_train, x_test, y_train, y_test = traintestset

    # Partial Least Squares (PLS) Method

    if optimise is True:
        # Creating initial values
        print(' ')
        print('    --- Commencing Optimisation of Regression Model ---')
        nlvs = np.zeros(imax - 1)
        rmsee = np.zeros(imax - 1)
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

                y_train_hat = pls_cv.predict(x_train).ravel()
                # rmse_train[j] = np.sqrt(mean_squared_error(y_train, y_train_hat))
                rmse_train[j] = np.sqrt(np.mean(np.square(np.subtract(y_train, y_train_hat))))
                j += 1

            # Gathering Statistic
            nlvs[i - 1] = i
            rmsee[i - 1] = np.mean(rmse_train)
            rmsecv[i - 1] = np.mean(rmse_test)

        # Implementing optimised parameters
        optstats = [nlvs, rmsecv, rmsee]
        lvs = 5  # some code to find knee joint
        print('    Optimised Lvs: {}'.format(lvs))
        print('    ------------ Completion of Optimisation -----------')
        print(' ')
    else:
        optstats = None
        lvs = 5

    print('The number of nLVs used is {}'.format(lvs))
    pls = PLSRegression(n_components=lvs)
    pls.fit(x_train, y_train)
    y_hat_test = pls.predict(x_test)
    y_hat_train = pls.predict(x_train)

    return pls, [y_train, y_hat_train], [y_test, y_hat_test], optstats


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

            opt_gbr = GradientBoostingRegressor(n_estimators=n_est,
                                                min_samples_split=min_sam,
                                                learning_rate=lr,
                                                max_depth=max_depth)

            # K-Fold object
            kfold = KFold(n_splits=n_splits)

            # Scoring object
            def rmse(y_true, y_pred):
                return np.sqrt(mean_squared_error(y_true, y_pred))

            scorer = make_scorer(rmse, greater_is_better=False)
            # CV score
            score = cross_val_score(opt_gbr, x_train, y_train, cv=kfold, scoring=scorer)

            return np.mean(score)

        # Creating bounds
        n_est_min, n_est_max = 100, 1000
        min_sam_min, min_sam_max = 5, 50
        lr_min, lr_max = 0.01, 0.1
        max_depth_min, max_depth_max = 1, 5
        bounds = Bounds([np.log(n_est_min), np.log(min_sam_min), np.sqrt(lr_min), np.log(max_depth_min)],
                        [np.log(n_est_max), np.log(min_sam_max), np.sqrt(lr_max), np.log(max_depth_max)])

        # Pre-loading initial values
        n_est0 = np.log(rand.uniform(n_est_min, n_est_max))
        min_sam0 = np.log(rand.uniform(min_sam_min, min_sam_max))
        lr0 = np.sqrt(rand.uniform(lr_min, lr_max))
        max_depth0 = np.log(rand.uniform(max_depth_min, max_depth_max))
        initial = np.array([n_est0, min_sam0, lr0, max_depth0])
        print('    ---------------- Initial Parameters ---------------')
        print('    n_estimators: {:.0f}\n'
              '    min_sample_split: {:.0f}\n'
              '    learning_rate: {:.2f} \n'
              '    max_depth: {:.0f}'
              .format(np.exp(n_est0), np.exp(min_sam0), np.square(lr0), np.exp(max_depth0))
              )
        print('    ---------------------------------------------------')

        # Begin Optimisation
        opt = minimize(fun, initial, method='trust-constr', bounds=bounds, options={'maxiter': 100})
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
              '    Final CV-MSE: {:.2f}'
              .format(np.exp(opt.x[0]), np.exp(opt.x[1]), np.square(opt.x[2]), np.exp(opt.x[3]), -opt.fun)
              )
        print('    ---------------------------------------------------')
        print('    ------------ Completion of Optimisation -----------')
        print(' ')

    gbr.fit(x_train, y_train)
    y_hat_train = gbr.predict(x_train)
    y_hat_test = gbr.predict(x_test)

    return gbr, [y_train, y_hat_train], [y_test, y_hat_test], None


def regress_plot(teststat, trainstat=None, optstat=None):
    y_test, y_hat_test = teststat

    # Model Statistics
    r2 = r2_score(y_test, y_hat_test)
    test_residue = y_hat_test.ravel() - y_test

    # residue plot
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, test_residue, c='r', edgecolors='k', label='Test Set')
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
    ax2.scatter(y_test, y_hat_test, c='r', edgecolors='k', label='Test Set')
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
    ax2.text(0, 0, '$R^2$= {:.2f}'.format(r2))

    if trainstat is not None:
        [y_train, y_hat_train] = trainstat

        train_residue = y_hat_train.ravel() - y_train
        ax1.scatter(y_train, train_residue, c='b', edgecolors='k', label='Train Set')
        ax1.legend()

        ax2.scatter(y_train, y_hat_train, c='b', edgecolors='k', label='Train Set')
        ax2.legend()

    if optstat is not None:
        [x_values, rmsecv, rmsee] = optstat

        # LV optimisation plot
        fig3, ax3 = plt.subplots()
        ax3.plot(x_values, rmsecv, c='b', label='RMSECV')
        ax3.plot(x_values, rmsee, c='r', label='RMSEE')
        ax3.set_xticks(x_values)
        ax3.set_xlabel('Number of LVs')
        ax3.set_ylabel('Error')
        ax3.set_title('Optimisation of LVs')
        ax3.legend()

    print('------------------Plots Generated------------------')
    plt.show()


def add_error(optpls, data, scaleddata, traindata=None):
    # addition of error into data
    x_data_reg = scaleddata[0][['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = scaleddata[1]['tR / min'].values
    y_hat_all = optpls.predict(x_data_reg)

    re = 100 * np.abs((y_hat_all.ravel() - y_data) / y_data)

    data['error'] = re.ravel()

    if traindata is not None:
        # calculation of mre
        [y_train, y_hat_train] = traindata
        msre = np.square(100 * (y_hat_train - y_train) / y_train).mean()
        rmsre = np.sqrt(np.mean(msre))
        print('The training RMSRE is {:.2f}%'.format(rmsre))
        return data, rmsre
    else:
        return data
