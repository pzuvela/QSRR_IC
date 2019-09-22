# Importing packages
import pandas as pd
from preprocessing import labelling, splitting, data_restrict
from classification import classify, classify_stats, classify_plot
from regression import regress_pls, regress_plot, add_error, regress_gbr, regress_stats


def traintest(modeldata, limxc, limdelc, n_splits, max_component, text=False):
    if text is True:
        print('### Commencing Training and Testing Procedure ####\n\n')

    # label generation with input limits
    df = labelling(modeldata, limxc, limdelc, method='delc')
    scaled_data, labelset, trset, sc, tr_max = splitting(df, 'Sequest')

    # classification model
    if text is True:
        print('############### Sequest Modelling ################')
    clf, clf_traindata, clf_testdata, clf_optdata = classify(labelset, n_splits, optimise=False)
    classify_stats(clf_traindata, clf_testdata, text=text)
    classify_plot(clf_testdata, clf_optdata)
    if text is True:
        print('############ End of Sequest Modelling ############\n')

    # regression model
    if text is True:
        print('################## QSRR Modelling ################')
    reg, reg_traindata, reg_testdata, reg_optdata = regress_pls(trset, n_splits, max_component)
    # reg, reg_traindata, reg_testdata, reg_optdata = regress_gbr(trset, n_splits, optimise=False)
    df2, mre = add_error(reg, df, scaled_data, reg_traindata)
    regress_stats(reg_traindata, reg_testdata, text=text)
    regress_plot(reg_testdata, reg_traindata, reg_optdata)
    if text is True:
        print('############# End of QSRR Modelling ##############\n\n')

    # label generation with regression stats
    df2 = labelling(df2, limxc, limdelc, mre, method='mre')

    # combination of above two model
    if text is True:
        print('############ Sequest + QSRR Modelling ############')
    scaled_data2, labelset2, trset2, sc2, tr_max2 = splitting(df2, 'QSRR')

    # classification with new labels
    clf2, clf_traindata2, clf_testdata2, clf_optdata2 = classify(labelset2, n_splits)
    classify_stats(clf_traindata2, clf_testdata2, text=text)
    classify_plot(clf_testdata2, clf_optdata2)
    if text is True:
        print('######### End of Sequest + QSRR Modelling #########\n\n')

    if text is True:
        print('## Completion of Training and Testing Procedure ##\n\n\n')

    return sc, sc2, clf, clf2, reg, mre, tr_max, reg_traindata


def validate(validationdata, models, limxc, limdelc, text=False):
    if text is True:
        print('######## Commencing Validation Procedure #########\n\n')

    sc, sc2, clf, clf2, reg, mre, tr_max, reg_traindata = models

    df_valid = labelling(validationdata, limxc, limdelc, method='delc')

    x_data = df_valid[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                       'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = df_valid[['tR / min', 'labels']]

    # Scaling
    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns)

    # Restricting validation set
    if text is True:
        print('Initial shape of validation Set : {}'.format(df_valid.shape))
    df_valid, x_data, y_data = data_restrict([df_valid, x_data, y_data], [tr_max, reg_traindata], ad=False)
    if text is True:
        print('Final shape of validation Set : {}'.format(df_valid.shape))

    # Splitting x and y data for clf and reg models
    x_data_clf = x_data[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp']]
    x_data_reg = x_data[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]

    y_data_clf = y_data['labels']
    y_data_reg = y_data['tR / min']

    if text is True:
        print('############### Sequest Modelling ################')
    y_hat_clf1 = clf.predict(x_data_clf)
    y_hat_proba1 = clf.predict_proba(x_data_clf)[:, 1]

    classify_stats(None, [x_data_clf, y_data_clf, y_hat_clf1], text=text)
    classify_plot([x_data_clf, y_data_clf, y_hat_clf1], [None, y_hat_proba1])
    if text is True:
        print('############ End of Sequest Modelling ############\n\n')

    if text is True:
        print('################## QSRR Modelling ################')
    y_hat_reg = reg.predict(x_data_reg).ravel()
    df_valid2 = add_error(reg, df_valid, [x_data, y_data])

    regress_stats(None, [x_data_reg, y_data_reg, y_hat_reg],text=text)
    regress_plot([x_data_reg, y_data_reg, y_hat_reg], reg_traindata)
    if text is True:
        print('############# End of QSRR Modelling ##############\n\n')

    df_valid2 = labelling(df_valid2, limxc, limdelc, mre, method='mre')

    x_data2 = df_valid2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                         'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                         'error']]
    # Scaling
    x_data2 = pd.DataFrame(sc2.transform(x_data2), columns=x_data2.columns)

    # Splitting X and Y data for clf2 model
    x_data_clf2 = x_data2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                           'error']]
    y_data_clf2 = df_valid2['labels']

    if text is True:
        print('############ Sequest + QSRR Modelling ############')
    y_hat_clf2 = clf2.predict(x_data_clf2)
    y_hat_proba2 = clf2.predict_proba(x_data_clf2)[:, 1]

    classify_stats(None, [x_data_clf2, y_data_clf2, y_hat_clf2], text=text)
    classify_plot([x_data_clf2, y_data_clf2, y_hat_clf2], [None, y_hat_proba2])
    if text is True:
        print('######### End of Sequest + QSRR Modelling #########\n\n')

    if text is True:
        print('####### Completion of Validation Procedure ########\n\n\n')

