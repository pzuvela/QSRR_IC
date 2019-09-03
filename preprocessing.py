import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

    def lambda3(charge, xc, delc, re, mean_re):
        if ((charge == 1 and xc >= lim_xc[0]) or (charge == 2 and xc >= lim_xc[1])
                or (charge >= 3 and xc >= lim_xc[2])) and delc >= lim_delc and re < mean_re:
            return 1
        else:
            return 0

    if method == 'xc':
        data['labels'] = data.apply(lambda x: lambda1(x['Charge'], x['XC']), axis=1)

    elif method == 'delc':
        data['labels'] = data.apply(lambda x: lambda2(x['Charge'], x['XC'], x['Delta Cn']), axis=1)

    elif method == 'mre' and mre is not None:
        data['labels'] = data.apply(lambda x: lambda3(x['Charge'], x['XC'], x['Delta Cn'], x['re'], mre), axis=1)
    else:
        print("Unrecognised Method: Choose from 'xc', 'delc' or 'mre'")

    return data


def splitting(rawdata, type):
    if type == 'Sequest':
        xdata = rawdata[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                     'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                     'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']].values
    elif type == 'QSRR':
        xdata = rawdata[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                         'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                         're']].values
    ydata = rawdata[['tR / min', 'labels']]
    # splitting
    x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3)

    # standardisation
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train_unscaled)
    x_test = sc.transform(x_test_unscaled)
    xdata = sc.transform(xdata)

    # splitting ydata
    y_label_train = y_train['labels']
    y_label_test = y_test['labels']
    y_tr_train = y_train['tR / min']
    y_tr_test = y_test['tR / min']

    # convert all to np.array
    y_label_train = y_label_train.values
    y_label_test = y_label_test.values
    y_tr_train = y_tr_train.values
    y_tr_test = y_tr_test.values


    return [[xdata, ydata],
            [x_train, x_test, y_label_train, y_label_test],
            [x_train, x_test, y_tr_train, y_tr_test],
            sc]
    # [[scaled data], [for classify], [for regress]]
