import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_mre(y_data, y_hat):
    return 100 * np.abs((y_hat.ravel() - y_data) / y_data).mean()


def get_rmsre(y_data, y_hat):
    return np.sqrt(np.square(100 * (y_hat - y_data) / y_data).mean())


def get_rmse(y_data, y_hat):
    return np.sqrt(np.square(y_hat - y_data).mean())


def get_mcc(y_data, y_hat):
    tn, fp, fn, tp = confusion_matrix(y_data, y_hat).ravel()
    return ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5


def fileprint(string, directory):
    with open(directory, 'a') as f:
        print(string, file=f)


def add_error(reg, rawdata, scaleddata):
    # scaled data required as reg model worked with scaled data
    x_data = scaleddata[0][['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = scaleddata[1]['tR / min'].values

    y_hat = reg.predict(x_data).ravel()
    re = 100 * np.abs((y_hat - y_data) / y_data)
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


""" 
Deprecated functions:

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

"""
