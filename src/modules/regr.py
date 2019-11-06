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

from importlib import import_module
from src.modules.func import get_rmse
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from numpy import mean


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


class RegressionHyperParamOpt:
    """

    (Hyper-)parameter optimization using differential evolution & cross-validation

    Implented models:
    1) PLS (CV)
    2) Gradient Boosting (sklearn) (DE)
    3) Extreme Boosting (xgBoost) (DE)
    4) RFR (DE)
    5) AdaBoost (DE)

    """

    def __init__(self, method, x_train_opt, y_train_opt, x_test_opt, y_test_opt, n_splits, proc_i, results_dir, count):

        """
        ------ Class initializaton -------

        :param method:
        :param x_train_opt:
        :param y_train_opt:
        :param x_test_opt:
        :param y_test_opt:
        :param n_splits:
        :param proc_i:
        :param results_dir:
        :param count:
        """

        """

        ----- Attributes ------

        """

        # Model (method) attribute
        self.method = method

        # Data attributes
        self.x_train_opt, self.y_train_opt, self.x_test_opt, self.y_test_opt = x_train_opt, y_train_opt, x_test_opt, \
            y_test_opt

        # Fixed attributes
        self.n_splits, self.proc_i, self.results_dir, self.count = n_splits, proc_i, results_dir, count

        """
        
        ----- Importing modules & packages ------
        
        """
        # Importing scipy
        self.opt = import_module('scipy.optimize')

        # Importing numpy modules
        self.round, self.mean, self.zeros, self.sqrt = [getattr(import_module('numpy'), i) for i in ['round', 'mean',
                                                                                                     'zeros', 'sqrt']]

        # Importing sklearn modules
        (self.KFold, self.cross_val_score), (self.make_scorer, self.mean_squared_error) = [getattr(
            import_module('sklearn.model_selection'), i) for i in ['KFold', 'cross_val_score']], [getattr(
                import_module('sklearn.metrics'), j) for j in ['make_scorer', 'mean_squared_error']]

        # Importing time & date modules
        self.time, self.strftime, self.gmtime = [getattr(import_module('time'), i) for i in
                                                 ['time', 'strftime', 'gmtime']]

        self.datetime = getattr(import_module('datetime'), 'datetime')

        # Import RMSE & knee point functions
        self.get_rmse, self.get_rmsre, self.knee = [getattr(import_module('src.modules.func'), i) for i in [
            'get_rmse', 'get_rmsre', 'knee']]

        # Dictionary of models
        self.models = {'xgb': (['xgboost'], ['XGBRegressor']),
                       'gbr': (['sklearn.ensemble'], ['GradientBoostingRegressor']),
                       'rfr': (['sklearn.ensemble'], ['RandomForestRegressor']),
                       'ada': (['sklearn.ensemble'], ['AdaBoostRegressor']),
                       'pls': (['sklearn.cross_decomposition'], ['PLSRegression'])}

        # Import filtering module
        self.filter = getattr(import_module('fnmatch'), 'filter')

        # Model assertion
        assert method in self.models.keys(), '# Please enter either ''pls'', ''ada'', ''rfr'', ''xgb'', or ''gbr'' !'

        # Import model
        self.reg_opt = (getattr(import_module(*self.models[self.method][0]), *self.models[self.method][1]))

        # Dictionary of file types for each hyper-parameter
        self.params_ftypes = {'xgb': ({'n_estimators': int, 'learning_rate': float, 'max_depth': int}),
                              'gbr': ({'n_estimators': int, 'learning_rate': float, 'max_depth': int}),
                              'rfr': ({'n_estimators': int, 'max_depth': int, 'min_samples_leaf': int}),
                              'ada': ({'n_estimators': int, 'learning_rate': float})}

        # Empty dictionary of hyper-parameters for the objective function
        self.params_opt = {'xgb': ({'n_estimators': int(), 'learning_rate': float, 'max_depth': int()}),
                           'gbr': ({'n_estimators': int(), 'learning_rate': float, 'max_depth': int()}),
                           'rfr': ({'n_estimators': int(), 'max_depth': int(), 'min_samples_leaf': int()}),
                           'ada': ({'n_estimators': int(), 'learning_rate': float()})}

        # Empty dictionary of hyper-parameters for the final values
        self.params_final = {'xgb': ({'n_estimators': int(), 'learning_rate': float, 'max_depth': int()}),
                             'gbr': ({'n_estimators': int(), 'learning_rate': float, 'max_depth': int()}),
                             'rfr': ({'n_estimators': int(), 'max_depth': int(), 'min_samples_leaf': int()}),
                             'ada': ({'n_estimators': int(), 'learning_rate': float()})}

        # Dictionary of hyper-parameter ranges
        self.params_ranges = {'xgb':  ({'n_est_lb': 10, 'n_est_ub': 500, 'lr_lb': 0.1, 'lr_ub': 0.9,
                                        'max_depth_lb': 1, 'max_depth_ub': 5}),
                              'gbr':  ({'n_est_min': 10, 'n_est_max': 500, 'lr_min': 0.1, 'lr_max': 0.9,
                                        'max_depth_min': 1, 'max_depth_max': 5})}

    @staticmethod
    def obj_fun(x, opt_list):

        # Unpack list
        models, method, params_ftypes, params_opt, reg_opt, x_train_opt, y_train_opt, n_splits = opt_list

        # Iterate over the two hyper-parameter dictionaries
        obj_fun_counter = 0
        for key_param in params_ftypes[method]:
            params_opt[method][key_param] = params_ftypes[method][key_param](x[obj_fun_counter])
            obj_fun_counter += 1

        reg_opt = (getattr(import_module(*models[method][0]), *models[method][1]))

        opt_model = reg_opt(objective="reg:squarederror").set_params(**params_opt[method]) \
            if method == 'xgb' else reg_opt().set_params(**params_opt[method])

        # CV score
        scorer_ens_opt = make_scorer(get_rmse)
        score = cross_val_score(opt_model, x_train_opt, y_train_opt, cv=KFold(n_splits=n_splits),
                                scoring=scorer_ens_opt)

        return mean(score)

    def optimize(self):

        # Ensemble learners
        if self.method in self.params_opt.keys():

            # QSRR Optimisation
            print('    ----------------- QSRR Optimisation ---------------')

            # Initial Params
            params_init = self.reg_opt().get_params(self.reg_opt())

            # Fit initial model
            reg_de = self.reg_opt(objective="reg:squarederror") if self.method == 'xgb' else self.reg_opt()
            reg_de.fit(self.x_train_opt, self.y_train_opt)

            # Initial predictions
            y_hat_train_init = reg_de.predict(self.x_train_opt)
            rmse_train_init = self.get_rmse(self.y_train_opt, y_hat_train_init)
            y_hat_test_init = reg_de.predict(self.x_test_opt)
            rmse_test_init = self.get_rmse(self.y_test_opt, y_hat_test_init)
            regopt_start = self.time()

            toprint_init = '    --------------------- Initial Parameters ---------------------\n' \
                           '    {}: {}\n' \
                           '    Initial RMSEE: {:.2f}\n' \
                           '    Initial RMSEP: {:.2f}\n' \
                           '    ---------------------------------------------------------------' \
                           .format(list(self.params_opt[self.method].keys()),
                                   [params_init[i] for i in self.params_opt[self.method].keys()],
                                   rmse_train_init,
                                   rmse_test_init)
            print(toprint_init)

            # Creating bounds
            bounds_lb = [self.params_ranges[self.method][i] for i in self.filter(self.params_ranges[self.method].keys(),
                                                                                 '*lb*')]
            bounds_ub = [self.params_ranges[self.method][i] for i in self.filter(self.params_ranges[self.method].keys(),
                                                                                 '*ub*')]
            bounds = self.opt.Bounds(bounds_lb, bounds_ub)

            final_values = self.opt.differential_evolution(self.obj_fun, bounds, workers=self.proc_i,
                                                           updating='deferred', mutation=(1.5, 1.9), popsize=20,
                                                           args=([self.models, self.method, self.params_ftypes,
                                                                  self.params_opt, self.reg_opt, self.x_train_opt,
                                                                  self.y_train_opt, self.n_splits], ))

            # Iterate over the two hyper-parameter dictionaries
            fin_counter = 0
            for key in self.params_ftypes[self.method]:
                self.params_final[self.method][key] = self.params_ftypes[self.method][key](final_values.x[fin_counter])
                fin_counter += 1

            params_final = self.params_final

            self.reg_opt.set_params(**self.params_final).fit(self.x_train_opt, self.y_train_opt)

            # Final Params
            y_hat_train_final = self.reg_opt.predict(self.x_train_opt)
            rmse_train_final = self.get_rmse(self.y_train_opt, y_hat_train_final)
            y_hat_test_final = self.reg_opt.predict(self.x_test_opt)
            rmse_test_final = self.get_rmse(self.y_test_opt, y_hat_test_final)
            regopt_time = self.time() - regopt_start

            toprint_final = '    ----------------------- Final Parameters ----------------------\n' \
                            '    {}\n' \
                            '    Final RMSECV: {:.3f}\n' \
                            '    Final RMSEE: {:.2f}' \
                            '    Final RMSEP: {:.2f}\n' \
                            '    Optimisation Duration: {}\n' \
                            '    ---------------------------------------------------------------' \
                            .format(params_final,
                                    final_values.fun,
                                    rmse_train_final,
                                    rmse_test_final,
                                    self.strftime("%H:%M:%S", self.gmtime(regopt_time)))

            print(toprint_final)

        elif self.method == 'pls':

            # Partial Least Squares (PLS)

            # Creating initial values
            print(' ')
            print('    --- Commencing Optimisation of PLS Model ---')

            imax = 10

            nlvs = self.zeros(imax)
            rmsecv = self.zeros(imax)
            for i in range(1, imax + 1):

                # KFold object
                kf = self.KFold(n_splits=self.n_splits)

                # Initiate cross validation (CV) model
                self.reg_opt().set_params(n_components=i)
                self.reg_opt().fit(self.x_train_opt, self.y_train_opt)

                # Pre-loading variables
                j = 0
                # Construct Scoring Object
                scorer_pls_opt = self.make_scorer(self.get_rmsre, greater_is_better=False)
                rmse_test = self.cross_val_score(self.reg_opt(), self.x_train_opt, self.y_train_opt,
                                                 cv=self.KFold(n_splits=self.n_splits), scoring=scorer_pls_opt)
                for train_i, test_i in kf.split(self.x_train_opt, self.y_train_opt):
                    # Defining the training set
                    x_train_cv = self.x_train_opt[train_i]
                    y_train_cv = self.y_train_opt[train_i]
                    # Defining the left out set
                    x_test_cv = self.x_train_opt[test_i]
                    y_test_cv = self.y_train_opt[test_i]
                    # Build PLS Model
                    reg_cv = self.reg_opt().fit(x_train_cv, y_train_cv)
                    # Generating RMSE
                    y_test_hat_cv = reg_cv.predict(x_test_cv).ravel()
                    rmse_test[j] = self.get_rmsre(y_test_cv, y_test_hat_cv)
                    rmse_test[j] = (self.sqrt(self.mean_squared_error(y_test_cv, y_test_hat_cv)))
                    j += 1
                # Gathering Statistics
                nlvs[i - 1] = i
                rmsecv[i - 1] = self.mean(rmse_test)
            lvs = self.knee(nlvs, rmsecv)
            toprint_final = '    Optimised n(LVs): {} \n' \
                            '    RMSECV: {:.3f} \n' \
                            '    ------------ Completion of Optimisation ----------- \n'.format(lvs, rmsecv[lvs - 1])

            print(toprint_final)

            params_final = {'n_components': nlvs}

        else:

            raise ValueError('# Please enter either ''pls'',''ada'',''rfr'', ''xgb'', or ''gbr''')

        with open(self.results_dir + '2019-QSRR_IC_PartIV-{}_{}_opt_run_{}.txt'.format(
                self.datetime.now().strftime('%d_%m_%Y-%H_%M'), self.method, self.count), "w") as text_file:
            text_file.write(toprint_final)

        return params_final
