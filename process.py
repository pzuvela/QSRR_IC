# Importing packages
import pandas as pd
from preprocessing import labelling, splitting, feature_selection
from classification import classify, classify_stats, classify_plot
from regression import regress_pls, regress_plot, add_error, regress_gbr


def traintest(modeldata, limxc, limdelc, n_splits, max_component):
    print('### Commencing Training and Testing Procedure ####')
    print('')

    # label generation with input limits
    df = labelling(modeldata, limxc, limdelc, method='delc')
    scaled_data, labelset, trset, sc, tr_max = splitting(df, 'Sequest')

    # classification model
    print('############### Sequest Modelling ################')
    clf, clf_optstats = classify(labelset, n_splits, optimise=False)
    classify_stats(clf_optstats)
    labeltestset = labelset[1], labelset[3]
    classify_plot(clf, labeltestset)
    print('############ End of Sequest Modelling ############')
    print('')

    # regression model
    print('################## QSRR Modelling ################')
    reg, reg_traindata, reg_testdata, reg_optstats = regress_pls(trset, n_splits, max_component, optimise=True)
    # reg, reg_traindata, reg_testdata, reg_optstats = regress_gbr(trset, n_splits, optimise=False)
    df2, mre = add_error(reg, df, scaled_data, reg_traindata)
    regress_plot(reg_testdata, reg_traindata, reg_optstats)
    print('############# End of QSRR Modelling ##############')
    print('')

    # label generation with regression stats
    df2 = labelling(df2, limxc, limdelc, mre, method='mre')

    # combination of above two model
    print('############ Sequest + QSRR Modelling ############')
    scaled_data2, labelset2, trset2, sc2, tr_max2 = splitting(df2, 'QSRR')

    # classification with new labels
    clf2, clf_optstats2 = classify(labelset2, n_splits)
    classify_stats(clf_optstats2)
    labeltestset2 = labelset2[1], labelset2[3]
    classify_plot(clf2, labeltestset2)
    print('######### End of Sequest + QSRR Modelling #########')
    print('')

    print('## Completion of Training and Testing Procedure ##')
    print('')
    print('')
    print('')
    return sc, sc2, clf, clf2, reg, mre, tr_max, reg_traindata


def validate(validationdata, models, limxc, limdelc):
    print('######## Commencing Validation Procedure #########')
    print('')

    sc, sc2, clf, clf2, reg, mre, tr_max, reg_traindata = models

    df_valid = labelling(validationdata, limxc, limdelc, method='delc')

    x_data = df_valid[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                       'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = df_valid[['tR / min', 'labels']]

    # Scaling
    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns)

    # Restricting validation set
    print('Initial shape of validation Set : {}'.format(df_valid.shape))
    df_valid, x_data, y_data = feature_selection([df_valid, x_data, y_data], [tr_max, reg_traindata], ad=False)
    print('Final shape of validation Set : {}'.format(df_valid.shape))

    # Splitting x and y data for clf and reg models
    x_data_clf = x_data[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp']]
    x_data_reg = x_data[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]

    y_data_clf = y_data['labels']
    y_data_reg = y_data['tR / min']

    print('############### Sequest Modelling ################')
    y_hat_clf1 = clf.predict(x_data_clf)
    classify_stats([y_data_clf, y_hat_clf1])
    classify_plot(clf, [x_data_clf, y_data_clf])
    print('############ End of Sequest Modelling ############')
    print('')

    print('################## QSRR Modelling ################')
    y_hat_reg = reg.predict(x_data_reg)
    df_valid2 = add_error(reg, df_valid, [x_data, y_data])
    regress_plot([x_data_reg, y_data_reg, y_hat_reg], reg_traindata)
    print('############# End of QSRR Modelling ##############')
    print('')

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

    print('############ Sequest + QSRR Modelling ############')
    y_hat_clf2 = clf2.predict(x_data_clf2)
    classify_stats([y_data_clf2, y_hat_clf2])
    classify_plot(clf2, [x_data_clf2, y_data_clf2])
    print('######### End of Sequest + QSRR Modelling #########')
    print('')

    print('####### Completion of Validation Procedure ########')
    print('')
    print('')
    print('')
