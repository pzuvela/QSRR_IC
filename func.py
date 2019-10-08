import numpy as np
import pandas as pd
from scipy.stats import norm, mode
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
    return fig


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


def updatetable(main_data, limxc, limdelc, models, i):
    sc, sc2, clf, clf2, reg, mre, tr_max, reg_traindata = models

    df_yproba1 = pd.DataFrame()
    df_yhat = pd.DataFrame()
    df_yproba2 = pd.DataFrame()

    df = labelling(main_data, limxc, limdelc, method='delc')
    x_data = df[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                 'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = df[['tR / min', 'labels']]

    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns)

    x_data_clf = x_data[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp']]
    y_data_clf = y_data[['labels']].values.ravel()
    y_hat_proba1 = clf.predict_proba(x_data_clf)[:, 1].ravel()

    x_data_reg = x_data[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data_reg = y_data[['tR / min']].values.ravel()
    y_hat_reg = reg.predict(x_data_reg).ravel()

    df2 = add_error(reg, df, [x_data, y_data])
    df2 = labelling(df2, limxc, limdelc, mre=mre, method='mre')

    x_data2 = df2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                   'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                   'error']]

    x_data2 = pd.DataFrame(sc2.transform(x_data2), columns=x_data2.columns)

    x_data_clf2 = x_data2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                           'error']]
    y_data_clf2 = df2[['labels']].values.ravel()
    y_hat_proba2 = clf2.predict_proba(x_data_clf2)[:, 1].ravel()

    return [y_data_clf, y_hat_proba1], [y_data_reg, y_hat_reg], [y_data_clf2, y_hat_proba2]


def add_true_mean_std(y_true, df):
    stats = []
    for i in range(len(df.columns)):
        label = df.columns[i]
        col_values = np.sort(df[label].values)
        j = int(len(col_values) * 2.5 / 100)
        k = int(len(col_values) - j)
        mean = col_values.mean()
        lower_value = col_values[j]
        upper_value = col_values[k]
        lower_limit = mean - lower_value
        upper_limit = upper_value - mean
        stats.append([mean, lower_limit, upper_limit, lower_value, upper_value])
    stats = np.transpose(stats)
    indices = ['mean', 'lower_limit', 'upper_limit', 'lower_value', 'upper_value']
    if y_true is not None:
        stats = np.vstack((y_true, stats))
        indices = ['actual', 'mean', 'lower_limit', 'upper_limit', 'lower_value', 'upper_value']
    df_stats = pd.DataFrame(stats, index=indices, columns=df.columns)
    df = pd.concat([df_stats, df])
    return df


def add_mean_std(df):
    mean = df.mean(axis=0).ravel()
    std = df.std(axis=0).ravel()
    stats = [mean, std]
    df_stats = pd.DataFrame(stats, index=['mean', 'std'])
    df = pd.concat([df_stats, df])
    return df


def get_limits(file):
    df = pd.read_csv(file)
    stats_list = []

    for i in range(1, len(df.columns)):
        label = df.columns[i]
        col_values = np.sort(df[label].values)
        j = int(len(col_values) * 2.5 / 100)
        k = int(len(col_values) - j)
        mean = col_values.mean()
        lower_value = col_values[j]
        upper_value = col_values[k]
        lower_limit = mean - lower_value
        upper_limit = upper_value - mean
        stats_list.append([label, mean, lower_limit, upper_limit, lower_value, upper_value])
        towrite = '{} limits are {:.2f}(+{:.2f};-{:.2f})'.format(label, mean, upper_limit, lower_limit)
        print(towrite)
    pd.DataFrame(stats_list, columns=['label', 'mean', 'lower_limit', 'upper_limit', 'lower_value', 'upper_value'])\
        .to_csv('{}_stats.csv'.format(file[:-4]), index=False)
