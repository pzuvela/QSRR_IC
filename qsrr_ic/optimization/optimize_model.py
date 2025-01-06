from fnmatch import filter
from typing import Optional

from qsrr_ic.models.qsrr import QsrrModel
from qsrr_ic.models.qsrr.domain_models import QsrrData
from qsrr_ic.optimization.domain_models import (
    HyperParameter,
    HyperParameterRegistry,
    OptimizerSettings,
    OptimizerResults,
)


# Dictionary of hyper-parameter ranges
self.params_ranges = {'xgb': ({'n_est_lb': 10, 'n_est_ub': 500, 'lr_lb': 0.1, 'lr_ub': 0.9,
                               'max_depth_lb': 1, 'max_depth_ub': 5}),
                      'gbr': ({'n_est_min': 10, 'n_est_max': 500, 'lr_min': 0.1, 'lr_max': 0.9,
                               'max_depth_min': 1, 'max_depth_max': 5}),
                      'ada': ({'n_est_lb': 300, 'n_est_ub': 1000, 'lr_lb': 0.1, 'lr_ub': 0.9}),
                      'rfr': ({'n_est_lb': 100, 'n_est_ub': 500, 'max_depth_lb': 8, 'max_depth_ub': 20,
                               'min_samples_lb': 1, 'min_samples_ub': 4}),
                      }

# Empty dictionary of hyper-parameters for the objective function
self.params_opt = {'xgb': ({'n_estimators': int(), 'learning_rate': float(), 'max_depth': int()}),
                   'gbr': ({'n_estimators': int(), 'learning_rate': float(), 'max_depth': int()}),
                   'rfr': ({'n_estimators': int(), 'max_depth': int(), 'min_samples_leaf': int()}),
                   'ada': ({'n_estimators': int(), 'learning_rate': float()})}




class QsrrModelOptimizer:

    def __init__(
        self,
        optimizer_settings: OptimizerSettings,
        qsrr_train_data: QsrrData,
        qsrr_test_data: Optional[QsrrData] = None
    ):

        self.optimizer_settings = optimizer_settings
        self.qsrr_train_data = qsrr_train_data
        self.qsrr_test_data = qsrr_test_data

        self.optimal_hyper_parameters: HyperParameterRegistry = HyperParameterRegistry()


    def objective_function(self, params: ndarray):

        # Iterate over the two hyper-parameter dictionaries
        obj_fun_counter = 0
        for key_param in params_ftypes[method]:
            params_opt[method][key_param] = params_ftypes[method][key_param](x[obj_fun_counter])
            obj_fun_counter += 1

        model = QsrrModel(
            regressor_type=self.optimizer_settings.regressor_type,
            qsrr_train_data=self.qsrr_train_data,
            qsrr_test_data=self.qsrr_test_data,
            hyper_parameters=hyper_parameters
        )

        reg_opt = (getattr(import_module(*models[method][0]), *models[method][1]))

        opt_model = reg_opt(objective="reg:squarederror").set_params(**params_opt[method]) \
            if method == 'xgb' else reg_opt(loss='exponential').set_params(**params_opt[method]) \
            if method == 'ada' else reg_opt().set_params(**params_opt[method])

        # CV score
        scorer_ens_opt = make_scorer(rmse_scorer)
        score = cross_val_score(opt_model, x_train_opt, y_train_opt, cv=KFold(n_splits=n_splits),
                                scoring=scorer_ens_opt)

        return mean(score)

    def optimize(self):

        # Ensemble learners
        if self.method in self.params_opt.keys():

            # QSRR Optimisation
            print('    ----------------- QSRR Optimisation ---------------')

            # Initial Params
            params_init = self.reg_opt().get_params()

            # Fit initial model
            reg_de = self.reg_opt(objective="reg:squarederror") if self.method == 'xgb' \
                else self.reg_opt(loss='exponential') if self.method == 'ada' else self.reg_opt()
            reg_de.fit(self.x_train_opt, self.y_train_opt)

            # Initial predictions
            y_hat_train_init = reg_de.predict(self.x_train_opt)
            y_hat_test_init = reg_de.predict(self.x_test_opt)
            rmse_train_init = rmse_scorer(self.y_train_opt, y_hat_train_init)
            rmse_test_init = rmse_scorer(self.y_test_opt, y_hat_test_init)
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
                                                                  self.y_train_opt, self.n_splits],))

            # Iterate over the two hyper-parameter dictionaries
            fin_counter = 0
            for key in self.params_ftypes[self.method]:
                self.params_final[self.method][key] = self.params_ftypes[self.method][key](final_values.x[fin_counter])
                fin_counter += 1

            self.params_final = self.params_final[self.method]  # Bug fix

            reg_de.set_params(**self.params_final).fit(self.x_train_opt, self.y_train_opt)  # Bug fix

            # Final Params
            y_hat_train_final = reg_de.predict(self.x_train_opt)
            rmse_train_final = rmse_scorer(self.y_train_opt, y_hat_train_final)
            y_hat_test_final = reg_de.predict(self.x_test_opt)
            rmse_test_final = rmse_scorer(self.y_test_opt, y_hat_test_final)
            regopt_time = self.time() - regopt_start

            toprint_final = '    ----------------------- Final Parameters ----------------------\n' \
                            '    {}\n' \
                            '    Final RMSECV: {:.3f}\n' \
                            '    Final RMSEE: {:.2f}' \
                            '    Final RMSEP: {:.2f}\n' \
                            '    Optimisation Duration: {}\n' \
                            '    ---------------------------------------------------------------' \
                .format(self.params_final,
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
                reg_opt = self.reg_opt()
                reg_opt.set_params(n_components=i)
                reg_opt.fit(self.x_train_opt, self.y_train_opt)

                # Pre-loading variables
                j = 0
                # Construct Scoring Object
                scorer_pls_opt = self.make_scorer(rmse_scorer, greater_is_better=False)
                rmse_test = self.cross_val_score(reg_opt, self.x_train_opt, self.y_train_opt,
                                                 cv=self.KFold(n_splits=self.n_splits), scoring=scorer_pls_opt)
                for train_i, test_i in kf.split(self.x_train_opt, self.y_train_opt):
                    # Defining the training set
                    x_train_cv = self.x_train_opt[train_i]
                    y_train_cv = self.y_train_opt[train_i]
                    # Defining the left out set
                    x_test_cv = self.x_train_opt[test_i]
                    y_test_cv = self.y_train_opt[test_i]
                    # Build PLS Model
                    reg_cv = reg_opt.fit(x_train_cv, y_train_cv)
                    # Generating RMSE
                    y_test_hat_cv = reg_cv.predict(x_test_cv).ravel()
                    rmse_test[j] = rmse_scorer(y_test_cv, y_test_hat_cv)
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

            self.params_final = {'n_components': lvs}
            self.opt_res = (nlvs, rmsecv)

        else:

            raise ValueError('# Please enter either ''xgb'', ''gbr'', ''ada'', ''rfr, or ''pls'' !')

        with open(self.results_dir + '2019-QSRR_IC_PartIV-{}_{}_opt_run_{}.txt'.format(
                self.datetime.now().strftime('%d_%m_%Y-%H_%M'), self.method, self.count), "w") as text_file:
            text_file.write(toprint_final)

        return self
