# -*- coding: utf-8 -*-
"""
Function for chemical domain of applicability and to construct the Williams plot

Input variables:
    1) y_train       : Observed values for the training set
    2) y_train_hat   : Predicted y-values for the training set
    3) y_test        : Observed values for the testing set
    4) y_test_hat    : Predicted y-values for the testing set
    5) X_train       : Descriptor values for the training set
    6) X_test        : Descriptor values for the testing set
    7) graph         : string to construct the Williams plot or not ('yes','no')

Author: Petar Zuvela

Date: 2019/06/19

Latest updates:
1) Fixed a bug in calculating leverages

"""

# Import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def ad(y1, y1_hat, y2, y2_hat, x1, x2, graph):

    # Mean centering the X-matrices
    # x1 = x1 - np.mean(x1, axis=0)
    # x2 = x2 - np.mean(x1, axis=0)

    # Compute the hat matrix
    h1, h2 = hat_matrix(x1, x2)

    # Compute the residuals
    r1, r2 = resids(y1, y1_hat, y2, y2_hat)

    # Total vector of standardized residuals
    res_scaled = np.concatenate((r1, r2), axis=0)

    # Total vector of leverages
    hat = np.array([np.concatenate((h1, h2), axis=0)]).T

    # Critical hat value
    k = np.size(x1, axis=1) + 1
    hat_star = 3 * (k / len(h1))

    # print('## Leverage & standardized residuals computed...')
    # print('## Critical leverage value is: ' + str("%.3f" % hat_star))

    # Plotting Williams plot
    if graph == 'yes':
        # print('## Constructing Williams plot...')
        # Warning limit for std residuals
        sigma = 3
        fig, ad_plot = plt.subplots()
        plt.scatter(h1, r1, c='C1', label='Training set')
        plt.scatter(h2, r2, c='C0', label='Testing set')
        plt.axhline(y=sigma, xmin=0, xmax=1, color='red', linestyle='dashed')
        plt.axhline(y=-sigma, xmin=0, xmax=1, color='red', linestyle='dashed')
        plt.axvline(x=hat_star, ymin=-sigma, ymax=sigma, color='red', linestyle='dashed')
        plt.xlabel("Leverage")
        plt.ylabel("Standardized residuals")
        plt.title("Applicability domain")
        ad_plot.legend()

    elif graph == 'no':
        pass
        # print('\n')
        # print('## Not constructing Williams plot...')
    return hat, res_scaled, hat_star


def hat_matrix(x1, x2):
    # Compute the hat matrix
    x1 = x1.values
    x2 = x2.values
    h_core = np.linalg.pinv(np.matmul(np.transpose(x1), x1))  # Core of the hat matrix
    h1_work = np.matmul(np.matmul(x1, h_core), np.transpose(x1))  # Training hat matrix
    h2_work = np.matmul(np.matmul(x2, h_core), np.transpose(x2))  # Testing hat matrix
    h1 = np.diag(h1_work)
    h2 = np.diag(h2_work)  # Diagonals of the hat matrices / leverages
    return h1, h2


def resids(y1, y1_hat, y2, y2_hat):

    # Computing the standardized residuals
    res1 = y1_hat - y1
    res2 = y2_hat - y2
    s1 = np.std(res1)  # Standard deviation of training residuals
    # s2 = np.std(res2)  # Standard deviation of testing residuals
    resid1 = np.divide(res1, s1)
    resid2 = np.divide(res2, s1)

    return resid1, resid2
