from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor


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
