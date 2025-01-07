from typing import (
    Optional,
    Union
)

import numpy as np
from numpy import ndarray

from scipy.optimize import (
    Bounds,
    differential_evolution
)


from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    LeaveOneOut
)

from qsrr_ic.metrics import Metrics
from qsrr_ic.models.qsrr import QsrrModel
from qsrr_ic.models.qsrr.domain_models import QsrrData
from qsrr_ic.models.qsrr.enums import RegressorType
from qsrr_ic.optimization.domain_models import (
    HyperParameter,
    HyperParameterRange,
    HyperParameterRegistry,
    OptimizerSettings,
    OptimizerResults,
)
from qsrr_ic.process.process_curve_data import ProcessCurveData
from qsrr_ic.optimization.enums import (
    CrossValidationType,
    HyperParameterName
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

        self.optimal_hyper_parameters: Optional[HyperParameterRegistry] = None

        cv_kwargs = {}

        if self.optimizer_settings.cv_settings.cv_type == CrossValidationType.KFold:
            cv_kwargs = {"n_splits": self.optimizer_settings.cv_settings.n_splits}

        self.cv: Union[KFold, LeaveOneOut] = self.optimizer_settings.cv_settings.cv_type.value(**cv_kwargs)

    def get_bounds(self) -> Bounds:

        bounds_lb = []
        bounds_ub = []

        for _, hp in self.optimizer_settings.hyper_parameter_ranges:
            bounds_lb.append(hp.lower)
            bounds_ub.append(hp.upper)

        return Bounds(bounds_lb, bounds_ub)

    def get_hyper_parameters_registry(self, hyper_parameters: ndarray[float]) -> HyperParameterRegistry:
        hyper_parameters_dict = {
            name: hyper_parameters[idx].item()
            for idx, name in enumerate(self.optimizer_settings.hyper_parameter_ranges.names())
        }
        return HyperParameterRegistry.from_dict(hyper_parameters_dict)

    def get_model(self, hyper_parameters: HyperParameterRegistry) -> QsrrModel:
        return QsrrModel(
            regressor_type=self.optimizer_settings.regressor_type,
            qsrr_train_data=self.qsrr_train_data,
            qsrr_test_data=self.qsrr_test_data,
            hyper_parameters=hyper_parameters
        )

    def cross_validate_model(self, qsrr_model: QsrrModel, greater_is_better: bool = True):
        score = cross_val_score(
            qsrr_model.model,
            self.qsrr_train_data.x,
            self.qsrr_train_data.y,
            cv=self.cv,
            scoring=make_scorer(Metrics.rmse, greater_is_better=greater_is_better)
        )
        return np.mean(score)

    def _objective_function(self, hyper_parameters: HyperParameterRegistry, greater_is_better: bool = True):
        qsrr_model = self.get_model(hyper_parameters)
        cv_score = self.cross_validate_model(qsrr_model, greater_is_better=greater_is_better)
        return cv_score

    def objective_function(self, hyper_parameters: ndarray[float]):
        hyper_parameters = self.get_hyper_parameters_registry(hyper_parameters)
        return self._objective_function(hyper_parameters)

    def _optimize_using_de(self):
        optimal_hyper_parameters = differential_evolution(
            self.objective_function,
            self.get_bounds(),
            workers=self.optimizer_settings.global_search_settings.n_jobs,
            updating='deferred',
            mutation=self.optimizer_settings.global_search_settings.mutation_rate,
            popsize=self.optimizer_settings.global_search_settings.population_size
        )
        self.optimal_hyper_parameters = self.get_hyper_parameters_registry(optimal_hyper_parameters.x)

    def _optimize_using_knee(self):

        if len(self.optimizer_settings.hyper_parameter_ranges) > 1:
            raise ValueError(
                "Only regressors with a single hyper-parameter can be optimized using knee point approach!"
            )

        name: HyperParameterName = self.optimizer_settings.hyper_parameter_ranges.names()[0]  # Only one parameter
        hp_range: HyperParameterRange = self.optimizer_settings.hyper_parameter_ranges.get(name)

        if hp_range.type() != int:
            raise ValueError("Knee point approach is currently only supported for int HPs!")

        min_: int = hp_range.lower.value
        max_: int = hp_range.upper.value

        x = []
        errors = []

        for value in range(min_, max_ + 1):
            hyper_parameters = HyperParameterRegistry()
            hyper_parameters.add(name, HyperParameter(value))
            cv_score = self._objective_function(hyper_parameters, greater_is_better=False)
            x.append(value)
            errors.append(cv_score)

        optimal_hp = ProcessCurveData.knee(np.array(x), np.array(errors))

        self.optimal_hyper_parameters = HyperParameterRegistry()
        self.optimal_hyper_parameters.add(name, HyperParameter(optimal_hp))

    def optimize(self):

        if self.optimizer_settings.regressor_type == RegressorType.PLS:
            self._optimize_using_knee()
        else:

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

            # Partial Least Squares (PLS)

            print(' ')
            print('    --- Commencing Optimisation of PLS Model ---')

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
