"""

Functions related to regression models
    1) regress_pls  : PLS modelling, training & prediction
    2) regress_gbr  : GBR modelling, training & prediction
    3) regress_xgbr : XGB modelling, training & prediction
    4) regress_rfr  : RFR modelling, training & prediction
    5) regress_ada  : ADB modelling, training & prediction
    4) optimization : Optimization of hyper-parameters for PLS, GBR and XGB

Notes:
    1) Optimization of the number of latent variables for PLS is performed using K-fold CV
    2) Hyper-parameter optimization for GBR and XGB is performed using differential evolution

"""


def regress_pls(trset, reg_params=None):

    from sklearn.cross_decomposition import PLSRegression

    x_train, x_test, y_train, y_test = trset

    # Default number of LVs
    model = PLSRegression(n_components=4)
    if reg_params is not None:
        model.set_params(**reg_params)
    model.fit(x_train, y_train)

    y_hat_test = model.predict(x_test).ravel()
    y_hat_train = model.predict(x_train).ravel()

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test]


def regress_gbr(trset, reg_params=None):

    from sklearn.ensemble import GradientBoostingRegressor

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

    from xgboost import XGBRegressor

    x_train, x_test, y_train, y_test = trset

    # eXtreme Gradient Boosting Regressor
    model = XGBRegressor(objective="reg:squarederror")
    if reg_params is not None:
        model.set_params(**reg_params)
    model.fit(x_train, y_train)

    y_hat_train = model.predict(x_train).ravel()
    y_hat_test = model.predict(x_test).ravel()

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test]


def regress_rfr(trset, reg_params=None):

    from sklearn.ensemble import RandomForestRegressor

    x_train, x_test, y_train, y_test = trset

    # Random Forest Regressor
    model = RandomForestRegressor()
    if reg_params is not None:
        model.set_params(**reg_params)
    model.fit(x_train, y_train)

    y_hat_train = model.predict(x_train).ravel()
    y_hat_test = model.predict(x_test).ravel()

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test]


def regress_ada(trset, reg_params=None):

    from sklearn.ensemble import AdaBoostRegressor

    x_train, x_test, y_train, y_test = trset

    # Adaptive Boosting Regressor
    model = AdaBoostRegressor()
    if reg_params is not None:
        model.set_params(**reg_params)
    model.fit(x_train, y_train)

    y_hat_train = model.predict(x_train).ravel()
    y_hat_test = model.predict(x_test).ravel()

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test]


def optimization(method, x_train_opt, y_train_opt, x_test_opt, y_test_opt, n_splits, proc_i, results_dir, count):

    """

    (Hyper-)parameter optimization using differential evolution & cross-validation

    Implented models:
    1) PLS (CV)
    2) Gradient Boosting (sklearn) (DE)
    3) Extreme Boosting (xgBoost) (DE)
    4) RFR (DE)
    5) AdaBoost (DE)

    """

    # Importing modules
    from scipy import optimize
    from time import time, strftime, gmtime
    from datetime import datetime
    from src.modules.func import get_rmse, knee
    from numpy import round, mean, zeros, sqrt
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer

    # Main conditionals
    if method in ['xbr', 'gbr', 'rfr', 'ada']:

        # xgBoost
        if method == 'xgb':

            from xgboost import XGBRegressor

            reg_opt = XGBRegressor(objective="reg:squarederror").fit(x_train_opt, y_train_opt)

        # sklearn gradient boosting
        elif method == 'gbr':

            from sklearn.ensemble import GradientBoostingRegressor

            reg_opt = GradientBoostingRegressor()
            reg_opt.fit(x_train_opt, y_train_opt)

        # sklearn rfr
        elif method == 'rfr':

            from sklearn.ensemble import RandomForestRegressor

            reg_opt = RandomForestRegressor()
            reg_opt.fit(x_train_opt, y_train_opt)

        # sklearn ada
        elif method == 'ada':

            from sklearn.ensemble import AdaBoostRegressor

            reg_opt = AdaBoostRegressor()
            reg_opt.fit(x_train_opt, y_train_opt)

        else:

            raise ValueError('# Please enter either ''pls'',''ada'',''rfr'', ''xgb'', or ''gbr''')

        # QSRR Optimisation
        print('    ----------------- QSRR Optimisation ---------------')

        # Initial Params
        initial = reg_opt.get_params()
        y_hat_train_init = reg_opt.predict(x_train_opt)
        initial_rmse_train = get_rmse(y_train_opt, y_hat_train_init)
        y_hat_test_init = reg_opt.predict(x_test_opt)
        initial_rmse_test = get_rmse(y_test_opt, y_hat_test_init)
        regopt_start = time()

        toprint = '    ---------------- Initial Parameters ---------------\n' \
                  '    n_estimators: {:.0f}\n' \
                  '    learning_rate: {:.2f} \n' \
                  '    max_depth: {:.0f}\n' \
                  '    Initial RMSEE: {:.2f}' \
                  '    Initial RMSEP: {:.2f}\n' \
                  '    ---------------------------------------------------' \
            .format(initial['n_estimators'],
                    initial['learning_rate'],
                    initial['max_depth'],
                    initial_rmse_train,
                    initial_rmse_test
                    )

        print(toprint)

        # Creating bounds
        n_est_min, n_est_max = 10, 500
        lr_min, lr_max = 0.01, 0.9
        max_depth_min, max_depth_max = 1, 5
        bounds = optimize.Bounds([n_est_min, lr_min, max_depth_min],
                                 [n_est_max, lr_max, max_depth_max])

        # Creating optimisation function, needs to be in each
        def reg_objective(x):
            # Descaling Parameters
            n_est = int(round(x[0], decimals=0))
            lr = x[1]
            max_depth = int(round(x[2], decimals=0))

            opt_model = reg_opt.set_params(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)

            # CV score
            scorer_ens_opt = make_scorer(get_rmse)
            score = cross_val_score(opt_model, x_train_opt, y_train_opt, cv=KFold(n_splits=n_splits),
                                    scoring=scorer_ens_opt)

            return mean(score)

        final_values = optimize.differential_evolution(reg_objective, bounds, workers=proc_i, updating='deferred',
                                                       mutation=(1.5, 1.9), popsize=20)
        reg_params = {'n_estimators': int(round(final_values.x[0], decimals=0)),
                      'learning_rate': final_values.x[1],
                      'max_depth': int(round(final_values.x[2], decimals=0))}

        reg_opt.set_params(**reg_params).fit(x_train_opt, y_train_opt)

        # Final Params
        y_hat = reg_opt.predict(x_train_opt)
        final_rmse_train = get_rmse(y_train_opt, y_hat)
        y_hat = reg_opt.predict(x_test_opt)
        final_rmse_test = get_rmse(y_test_opt, y_hat)
        regopt_time = time() - regopt_start

        toprint = '    ----------------- Final Parameters ----------------\n' \
                  '    n_estimators: {:.0f}\n' \
                  '    learning_rate: {:.2f} \n' \
                  '    max_depth: {:.0f}\n' \
                  '    Final RMSECV: {:.3f}\n' \
                  '    Final RMSEE: {:.2f}' \
                  '    Final RMSEP: {:.2f}\n' \
                  '    Optimisation Duration: {}\n' \
                  '    ---------------------------------------------------\n' \
            .format(int(round(final_values.x[0], decimals=0)),
                    final_values.x[1],
                    int(round(final_values.x[2], decimals=0)),
                    final_values.fun,
                    final_rmse_train,
                    final_rmse_test,
                    strftime("%H:%M:%S", gmtime(regopt_time))
                    )
        print(toprint)

        with open(results_dir + '2019-QSRR_IC_PartIV-{}_{}_opt_run_{}.txt'.format(
                datetime.now().strftime('%d_%m_%Y-%H_%M'), method, count), "w") as text_file:
            text_file.write(toprint)

    elif method == 'pls':
        # Partial Least Squares (PLS)

        from src.modules.func import get_rmsre
        from sklearn.metrics import mean_squared_error
        from sklearn.cross_decomposition import PLSRegression

        # Creating initial values
        print(' ')
        print('    --- Commencing Optimisation of PLS Model ---')

        imax = 10
        nlvs = zeros(imax)
        rmsecv = zeros(imax)
        for i in range(1, imax+1):

            # KFold object
            kf = KFold(n_splits=n_splits)

            # Initiate cross validation (CV) model
            reg_opt = PLSRegression(n_components=i)
            reg_opt.fit(x_train_opt, y_train_opt)

            # Pre-loading variables
            j = 0
            # Construct Scoring Object
            scorer_pls_opt = make_scorer(get_rmsre, greater_is_better=False)
            rmse_test = cross_val_score(reg_opt, x_train_opt, y_train_opt, cv=KFold(n_splits=n_splits),
                                        scoring=scorer_pls_opt)
            for train_i, test_i in kf.split(x_train_opt, y_train_opt):
                # Defining the training set
                x_train_cv = x_train_opt[train_i]
                y_train_cv = y_train_opt[train_i]
                # Defining the left out set
                x_test_cv = x_train_opt[test_i]
                y_test_cv = y_train_opt[test_i]
                # Build PLS Model
                reg_opt.fit(x_train_cv, y_train_cv)
                # Generating RMSE
                y_test_hat_cv = reg_opt.predict(x_test_cv).ravel()
                rmse_test[j] = get_rmsre(y_test_cv, y_test_hat_cv)
                rmse_test[j] = (sqrt(mean_squared_error(y_test_cv, y_test_hat_cv)))
                j += 1
            # Gathering Statistics
            nlvs[i - 1] = i
            rmsecv[i - 1] = mean(rmse_test)
        lvs = knee(nlvs, rmsecv)  # some code to find knee point
        toprint = '    Optimised n(LVs): {} \n' \
                  '    RMSECV: {:.3f} \n' \
                  '    ------------ Completion of Optimisation ----------- \n'.format(lvs, rmsecv[lvs-1])

        print(toprint)

        with open(results_dir + '2019-QSRR_IC_PartIV-{}_{}_opt_run_{}.txt'.format(
                datetime.now().strftime('%d_%m_%Y-%H_%M'), method, count), "w") as text_file:
            text_file.write(toprint)

        reg_params = {'n_components': lvs}

    else:

        raise ValueError('# Please enter either ''pls'',''ada'',''rfr'', ''xgb'', or ''gbr''')

    return reg_params
