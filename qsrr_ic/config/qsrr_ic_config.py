class Config:
    def __init__(
        self
    ):
        # Dictionary of hyper-parameter ranges
        self.params_ranges = {'xgb': ({'n_est_lb': 10, 'n_est_ub': 500, 'lr_lb': 0.1, 'lr_ub': 0.9,
                                       'max_depth_lb': 1, 'max_depth_ub': 5}),
                              'gbr': ({'n_est_min': 10, 'n_est_max': 500, 'lr_min': 0.1, 'lr_max': 0.9,
                                       'max_depth_min': 1, 'max_depth_max': 5}),
                              'ada': ({'n_est_lb': 300, 'n_est_ub': 1000, 'lr_lb': 0.1, 'lr_ub': 0.9}),
                              'rfr': ({'n_est_lb': 100, 'n_est_ub': 500, 'max_depth_lb': 8, 'max_depth_ub': 20,
                                       'min_samples_lb': 1, 'min_samples_ub': 4}),
                              }
