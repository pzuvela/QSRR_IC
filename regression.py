import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from petar.ad import ad


def regress_pls(trset, reg_params=None):
    x_train, x_test, y_train, y_test = trset

    # Partial Least Squares (PLS)
    """
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

            # Pre-loading variables
            rmse_train = np.zeros(n_splits)
            rmse_test = np.zeros(n_splits)
            j = 0

            # Construct Scoring Object
            scorer = make_scorer(func.get_rmsre, greater_is_better=False)

            rmse_test = cross_val_score(pls_cv, x_train, y_train, cv=KFold(n_splits=n_splits), scoring=scorer)

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
                rmse_test[j] = func.get_rmsre(y_test_cv, y_test_hat_cv)
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
        lvs = func.get_knee(nlvs, rmsecv)  # some code to find knee point
        print('    Optimised Lvs: {}'.format(lvs))
        print('    ------------ Completion of Optimisation -----------')
        print(' ')
    else:
        optstats = None
        lvs = 5
    """
    lvs = 5
    model = PLSRegression(n_components=lvs)
    if reg_params is not None:
        model.set_params(**reg_params)
    model.fit(x_train, y_train)

    y_hat_test = model.predict(x_test).ravel()
    y_hat_train = model.predict(x_train).ravel()

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test]


def regress_gbr(trset, reg_params=None):
    x_train, x_test, y_train, y_test = trset

    # Gradient Boosting Regressor
    model = GradientBoostingRegressor()
    if reg_params is not None:
        model.set_params(**reg_params)
    model.fit(x_train, y_train)

    y_hat_train = model.predict(x_train).ravel()
    y_hat_test = model.predict(x_test).ravel()

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test]


def regress_xgbr(trset, reg_params=None):
    x_train, x_test, y_train, y_test = trset

    # eXtreme Gradient Boosting Regressor
    model = XGBRegressor(objective="reg:squarederror")
    if reg_params is not None:
        model.set_params(**reg_params)
    model.fit(x_train, y_train)

    y_hat_train = model.predict(x_train).ravel()
    y_hat_test = model.predict(x_test).ravel()

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test]


def regress_plot(teststat, trainstat=None, optstat=None):
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

    fig7, ax7 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    if trainstat is not None:
        x_test, y_test, y_hat_test = teststat
        x_train, y_train, y_hat_train = trainstat
        y_hat_train = y_hat_train.ravel()

        train_r2 = r2_score(y_train, y_hat_train)
        train_residue = y_hat_train - y_train

        ad(y_train, y_hat_train, y_test, y_hat_test, x_train, x_test, 'yes')

        ax7.hist(train_residue, color='C0', alpha=0.7, density=True, label='Train Set')

        ax1.scatter(y_train, train_residue, alpha=0.7, c='C0', label='Train Set')
        ax1.legend()

        ax2.scatter(y_train, y_hat_train, alpha=0.7, c='C0', label='Train Set')
        ax2.text(0.8, 0.17, 'Training R2 = {:.2f}'.format(train_r2), horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.legend()

    x_test, y_test, y_hat_test = teststat
    y_hat_test = y_hat_test.ravel()

    test_residue = y_hat_test - y_test

    # res histogram
    ax7.hist(test_residue, color='C1', density=True, label='Test Set')
    ax7.set_ylabel('Instances')
    ax7.set_xlabel('Residual')
    ax7.set_title('Residual Histogram')

    # residue plot
    ax1.scatter(y_test, test_residue, c='C1', label='Test Set')
    lim1 = [np.min(ax1.get_xlim()), np.max(ax1.get_xlim())]
    lim2 = [0, 0]
    ax1.plot(lim1, lim2, c='k')
    ax1.set_xlim(lim1)
    ax1.set_xlabel('Retention Time')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residual of prediction')
    ax1.legend()

    # response plot
    test_r2 = r2_score(y_test, y_hat_test)
    ax2.scatter(y_test, y_hat_test, c='C1', label='Test Set')
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
    ax2.text(0.8, 0.07, 'Testing R2 = {:.2f}'.format(test_r2), horizontalalignment='center',
             verticalalignment='center', transform=ax2.transAxes)
    ax2.legend()

    return fig7, fig1, fig2, ax2
    # residual histogram, residual distribution, response plot
