from typing import (
    Dict,
    List
)

from qsrr_ic.config import QsrrIcConfig, HyperParameterConfig
from qsrr_ic.domain_models import HyperParameterRegistry
from qsrr_ic.enums import (
    RegressorType,
    TrainingType
)
from qsrr_ic.load import (
    load_dataset,
    QsrrIcData,
    QsrrIcDataset
)
from qsrr_ic.models.qsrr import QsrrModel
from qsrr_ic.models.qsrr.domain_models import QsrrData
from qsrr_ic.optimization import QsrrModelOptimizer
from qsrr_ic.optimization.domain_models import (
    OptimizerSettings,
    OptimizerResults
)
from qsrr_ic.process import ProcessData
from qsrr_ic.runners import (
    QsrrModelRunner,
    QsrrResamplingWithReplacementModelRunner,
    QsrrOptimizerRunner,
    QsrrIcModelRunner
)


def main(settings_path: str):

    # 1. Read settings
    config: QsrrIcConfig = QsrrIcConfig.from_json(settings_path)

    # 2. Load dataset
    dataset: QsrrIcDataset = load_dataset(config.dataset_config)

    # 3. Process dataset for training
    data: QsrrIcData = ProcessData(dataset).process()

    # 4. Prepare & split QSRR data to train/test
    qsrr_data: QsrrData = QsrrData(
        y=data.isocratic_retention,
        x=data.molecular_descriptors_for_qsrr_training
    )
    qsrr_train_data, qsrr_test_data = qsrr_data.split(config.train_test_split_config)

    # 5. Train QSRR models

    qsrr_models: Dict[RegressorType, QsrrModel] = {}

    hyper_parameter_configs: Dict[RegressorType, HyperParameterConfig] = {}

    for regressor_type, hyper_parameter_config in config.hyper_parameter_config.items():

        if config.training_type == TrainingType.Optimization:

            # Optimization Mode
            optimizer_settings = OptimizerSettings(
                regressor_type,
                hyper_parameter_config.hyper_parameter_registry,
                cv_settings=config.cross_validation_config.cv_settings,
                global_search_settings=config.global_search_config.global_search_settings
            )

            optimizer_runner = QsrrOptimizerRunner()

            optimizer: QsrrModelOptimizer
            optimizer_results: OptimizerResults

            optimizer, optimizer_results = optimizer_runner.run(
                optimizer_settings=optimizer_settings,
                qsrr_train_data=qsrr_train_data,
                qsrr_test_data=qsrr_test_data
            )

            qsrr_models[regressor_type] = optimizer_results.optimal_qsrr_model
            hyper_parameter_configs[regressor_type] = HyperParameterConfig(
                regressor_type=regressor_type,
                hyper_parameter_registry=optimizer_results.optimal_hyper_parameters
            )

        elif config.training_type == TrainingType.SingleTrain:

            # Single Training Mode

            model_runner = QsrrModelRunner()

            model: QsrrModel = model_runner.run(
                regressor_type=regressor_type,
                qsrr_train_data=qsrr_train_data,
                qsrr_test_data=qsrr_test_data,
                hyper_parameters=hyper_parameter_config.hyper_parameter_registry
            )

            qsrr_models[regressor_type] = model
            hyper_parameter_configs[regressor_type] = hyper_parameter_config

    # 6. Train IC models

    # 7. Resampling with replacement
    bootstrapped_models: Dict[RegressorType, List[QsrrModel]] = {}

    if config.resampling_with_replacement_config is not None \
        and config.resampling_with_replacement_config.use_resampling:

        for regressor_type, hyper_parameter_config in hyper_parameter_configs.items():
            model_runner = QsrrResamplingWithReplacementModelRunner()
            bootstrapped_models[regressor_type] = model_runner.run(
                regressor_type=regressor_type,
                config=config.resampling_with_replacement_config,
                hyper_parameter_config=hyper_parameter_config,
                train_test_config=config.train_test_split_config,
                qsrr_data=qsrr_data
            )

    # 8. Save results
    ...

if __name__ == "__main__":
    main()
