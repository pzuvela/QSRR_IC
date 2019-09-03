import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

# input from preprocessing
# splitting(df)[1] should be x_train ...


def regress(traintestset, n_splits, imax, optimise=False):
    # Partial Least Squares (PLS) Method
    x_train, x_test, y_train, y_test = traintestset

    if optimise is True:
        # Creating initial values
        print(' ')
        print('    --- Commencing Optimisation of Regression Model ---')
        x_values, rmsee, rmsecv = [], [], []
        for i in range(1, imax):
            # Initiate cross validation (CV) model
            pls_cv = PLSRegression(n_components=i)

            # K-Fold Object
            kf = KFold(n_splits=n_splits)

            # Pre-loading variables
            rmse_train, rmse_test = [], []

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
                y_train_hat_cv = pls_cv.predict(x_train_cv)
                y_test_hat_cv = pls_cv.predict(x_test_cv)
                rmse_train.append(np.sqrt(mean_squared_error(y_train_cv, y_train_hat_cv)))
                rmse_test.append(np.sqrt(mean_squared_error(y_test_cv, y_test_hat_cv)))

            # Gathering Statistic
            x_values.append(i)
            rmsee.append(np.mean(rmse_train))
            rmsecv.append(np.mean(rmse_test))
            optstats = [x_values, rmsecv, rmsee]

        # Implementing optimised parameters
        lvs = 5  # some code to find knee joint
        print('    Optimised Lvs: {}'.format(lvs))
        print('    ------------ Completion of Optimisation -----------')
        print(' ')

    lvs = 5
    print('The number of nLVs used is {}'.format(lvs))
    pls = PLSRegression(n_components=lvs)
    pls.fit(x_train, y_train)
    y_hat_test = pls.predict(x_test)
    y_hat_train = pls.predict(x_train)

    if optimise is True:
        return pls, [y_train, y_hat_train], [y_test, y_hat_test], optstats
    else:
        return pls, [y_train, y_hat_train], [y_test, y_hat_test]


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
        np.max([ax2.get_xlim(), ax2.get_ylim()])   # max of both axes
    ]
    ax2.plot(lims, lims, c='k')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('predicted')
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
        ax3.set_ylabel('Error / %')
        ax3.set_title('Optimisation of LVs')
        ax3.legend()

    print('------------------Plots Generated------------------')
    plt.show()


def add_re(optpls, data, scaleddata, traindata=None):
    # addition of re into data
    xdata = scaleddata[0]
    ydata = scaleddata[1]['tR / min'].values
    y_hat_all = optpls.predict(xdata)

    re = 100 * np.absolute(y_hat_all.ravel() - ydata) / ydata

    data['re'] = re.ravel()

    if traindata is not None:
        # calculation of mre
        [y_train, y_hat_train] = traindata
        train_re = 100 * np.absolute(y_hat_train - y_train) / y_train
        mre = np.mean(train_re)
        print('The training MRE is {:.2f}'.format(mre))
        return data, mre
    else:
        return data
