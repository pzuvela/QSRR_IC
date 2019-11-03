"""

(Hyper-)parameter optimization using differential evolution

Implented models:
1) PLS
2) Gradient Boosting (sklearn)
3) Extreme Boosting (xgBoost)

"""

# Import scipy optimize
from scipy import optimize

# xgBoost
if method == 'xgb':

    from xgboost import XGBRegressor

    reg_opt = XGBRegressor(objective="reg:squarederror").fit(x_train_opt, y_train_opt)

# sklearn gradient boosting
elif method == 'gbr':

    from sklearn.ensemble import GradientBoostingRegressor

    reg_opt = GradientBoostingRegressor()
    reg_opt.fit(x_train_opt, y_train_opt)

# PLS
elif method == 'pls':

    from sklearn.cross_decomposition import PLSRegression

    reg_opt = PLSRegression()
    reg_opt.fit(x_train_opt, y_train_opt)

# Default
else:
    from xgboost import XGBRegressor

    reg_opt = XGBRegressor(objective="reg:squarederror").fit(x_train_opt, y_train_opt)

if method == 'xgb' or 'gbr':
    # QSRR Optimisation
    print('    ----------------- QSRR Optimisation ---------------')

    # Initial Params
    initial = reg_opt.get_params()
    y_hat_opt = reg_opt.predict(x_train_opt)
    initial_rmse_train = get_rmse(y_train_opt, y_hat_opt)
    y_hat_opt = reg_opt.predict(x_test_opt)
    initial_rmse_test = get_rmse(y_test_opt, y_hat_opt)
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
        scorer = make_scorer(get_rmse)
        score = cross_val_score(opt_model, x_train_opt, y_train_opt, cv=KFold(n_splits=n_splits), scoring=scorer)

        return mean(score)


    final_values = optimize.differential_evolution(reg_objective, bounds, workers=proc_i, updating='deferred',
                                                   mutation=(1.5, 1.9), popsize=20)
    reg_params = {'n_estimators': int(round(final_values.x[0], decimals=0)),
                  'learning_rate': final_values.x[1],
                  'max_depth': int(round(final_values.x[2], decimals=0))}

