import numpy as np


def get_mre(y_data, y_hat):
    return 100 * np.abs((y_hat.ravel() - y_data) / y_data).mean()

def get_rmsre(y_data, y_hat)
    return np.sqrt(np.square(100 * (y_hat_test - y_test) / y_test).mean())