import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt
from preprocessing import labelling


def get_mre(y_data, y_hat):
    return 100 * np.abs((y_hat.ravel() - y_data) / y_data).mean()


def get_rmsre(y_data, y_hat):
    return np.sqrt(np.square(100 * (y_hat - y_data) / y_data).mean())


def histplot(y_data, title, x_axis):
    fig, ax = plt.subplots()
    ax.hist(y_data, bins=20, density=True)
    x_min, x_max = ax.get_xlim()
    y_mean, y_std = norm.fit(y_data)
    p = norm.pdf(np.linspace(x_min, x_max, 100), y_mean, y_std)
    ax.plot(np.linspace(x_min, x_max, 100), p)
    ax.set_title(title)
    ax.set_ylabel('Probability Density')
    ax.set_xlabel(x_axis)
    ax.text(0.2, 0.9, s='Mean:{:.2f} +/- {:.2f}'.format(y_mean, 3*y_std), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.5))


def add_error(reg, rawdata, scaleddata):
    # addition of error into data
    x_data_reg = scaleddata[0][['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = scaleddata[1]['tR / min'].values
    y_hat_all = reg.predict(x_data_reg)

    re = 100 * np.abs((y_hat_all.ravel() - y_data) / y_data)

    rawdata.loc[:, 'error'] = re.ravel()
    return rawdata


def knee(x, y):
    del1, del2 = [], []
    for i in range(0, len(x) - 1):
        del1.append(y[i + 1] - y[i])

    for j in range(0, len(del1) - 1):
        del2.append(del1[j + 1] - del1[j])

    max_del2 = np.argmax(del2)
    kneept = x[max_del2 + 2]
    return int(kneept)


def updatetable(df_tobeupdated, main_data, limxc, limdelc, models, i):
    sc, sc2, clf, clf2, reg, mre, tr_max, reg_traindata = models

    df = labelling(main_data, limxc, limdelc, method='delc')
    x_data = df[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                 'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = df[['tR / min', 'labels']]

    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns)

    x_data_clf = x_data[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp']]
    y_hat_proba1 = clf.predict_proba(x_data_clf)

    x_data_reg = x_data[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_hat_reg = reg.predict(x_data_reg).ravel()

    df2 = add_error(reg, df, [x_data, y_data])
    x_data2 = df2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                   'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                   'error']]

    x_data2 = pd.DataFrame(sc2.transform(x_data2), columns=x_data2.columns)

    x_data_clf2 = x_data2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                           'error']]
    y_hat_proba2 = clf2.predict_proba(x_data_clf2)

    col_name1 = 'iter{}_y_proba1'.format(i)
    col_name2 = 'iter{}_y_hat_reg'.format(i)
    col_name3 = 'iter{}_y_proba2'.format(i)

    newdf = df_tobeupdated.assign[{col_name1: y_hat_proba1, col_name2: y_hat_reg, col_name3: y_hat_proba2}]
    return newdf
