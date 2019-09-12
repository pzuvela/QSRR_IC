import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from petar.ad import hat_matrix


# labels generation
def labelling(data, lim_xc, lim_delc, mre=None, method='xc'):
    def lambda1(charge, xc):
        if (charge == 1 and xc >= lim_xc[0]) or (charge == 2 and xc >= lim_xc[1]) \
                or (charge >= 3 and xc >= lim_xc[2]):
            return 1
        else:
            return 0

    def lambda2(charge, xc, delc):
        if ((charge == 1 and xc >= lim_xc[0]) or (charge == 2 and xc >= lim_xc[1])
                or (charge >= 3 and xc >= lim_xc[2])) and delc >= lim_delc:
            return 1
        else:
            return 0

    def lambda3(charge, xc, delc, error, mean_error):
        if ((charge == 1 and xc >= lim_xc[0]) or (charge == 2 and xc >= lim_xc[1])
                or (charge >= 3 and xc >= lim_xc[2])) and delc >= lim_delc and error < mean_error:
            return 1
        else:
            return 0

    if method == 'xc':
        data['labels'] = data.apply(lambda x: lambda1(x['Charge'], x['XC']), axis=1)

    elif method == 'delc':
        data['labels'] = data.apply(lambda x: lambda2(x['Charge'], x['XC'], x['Delta Cn']), axis=1)

    elif method == 'mre' and mre is not None:
        data['labels'] = data.apply(lambda x: lambda3(x['Charge'], x['XC'], x['Delta Cn'], x['error'], mre), axis=1)
    else:
        print("Unrecognised Method: Choose from 'xc', 'delc' or 'mre'")

    return data


def splitting(data, type):
    if type == 'Sequest':
        x_data = data[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                       'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    elif type == 'QSRR':
        x_data = data[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                       'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                       'error']]

    y_data = data[['tR / min', 'labels']]

    # splitting
    x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)

    # standardisation
    sc = StandardScaler()
    x_train = pd.DataFrame(sc.fit_transform(x_train_unscaled), columns=x_train_unscaled.columns)
    x_test = pd.DataFrame(sc.transform(x_test_unscaled), columns=x_test_unscaled.columns)
    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns)

    # splitting xdata
    if type == 'Sequest':
        x_train_clf = x_train[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp']]
        x_test_clf = x_test[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp']]
        x_train_reg = x_train[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
        x_test_reg = x_test[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    elif type == 'QSRR':
        x_train_clf = x_train[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp', 'error']]
        x_test_clf = x_test[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp', 'error']]
        x_train_reg = x_train[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
        x_test_reg = x_test[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]

    # splitting y_data
    y_train_clf = y_train['labels'].values
    y_test_clf = y_test['labels'].values
    y_train_reg = y_train['tR / min'].values
    y_test_reg = y_test['tR / min'].values

    tr_max = np.max(y_train_reg)

    return [[x_data, y_data],
            [x_train_clf, x_test_clf, y_train_clf, y_test_clf],
            [x_train_reg, x_test_reg, y_train_reg, y_test_reg],
            sc, tr_max]
    # [[scaled data], [for classify], [for regress], scaler and tr_max for validation]


def feature_selection(df, trainstat, tr=True, ad=False):
    # NOTE THAT ALL DATA SHOULD BE SCALED PRIOR TO SUBMISSION TO THIS FUNCTION
    tr_max, reg_traindata = trainstat
    x_train, y_train, y_hat_train = reg_traindata
    df_valid, x_data, y_data = df  # df_valid is unscaled

    if tr is True:
        y_data_reg = y_data['tR / min']

        df_valid = df_valid.loc[y_data_reg < tr_max]
        x_data = x_data.loc[y_data_reg < tr_max]
        y_data = y_data.loc[y_data_reg < tr_max]

    if ad is True:
        x_data_reg = x_data[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]

        # need to find h for each data point
        h1, h2 = hat_matrix(x_train, x_data_reg)

        # need to find critical h of training set
        k = np.size(x_train, axis=1) + 1
        hat_star = 3 * (k / len(h1))

        selection = [True if i >= hat_star else False for i in h2]
        df_valid = df_valid[selection]
        x_data = x_data[selection]
        y_data = y_data[selection]

    return df_valid, x_data, y_data
