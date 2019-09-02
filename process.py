# Importing packages
from preprocessing import labelling, splitting
from classification import classify, classify_stats, add_status, classify_plot
from regression import regress, regress_plot, add_re


def traintest(modeldata, limxc, limdelc, n_splits, max_component):
    print('-----Commencing Training and Testing Procedure-----')
    print('')

    # label generation
    df = labelling(modeldata, limxc, limdelc, None, 'delc')
    scaled_data, labelset, trset, sc = splitting(df, 'Sequest')

    # classification model
    print('-----------------Sequest Modelling-----------------')
    clf, clfstats = classify(labelset)
    classify_stats(clfstats)
    labeltestset = labelset[1], labelset[3]
    classify_plot(clf, labeltestset)
    df = add_status(clf, df, scaled_data, 'Sequest')
    print('')

    # regression model
    print('-------------------QSRR Modelling------------------')
    optpls, plstraindata, plstestdata, optstats = regress(trset, n_splits, max_component)
    regress_plot(plstraindata, plstestdata, optstats)
    print('')

    print('--------------Sequest + QSRR Modelling-------------')
    # 2nd round classification
    df2, mre = add_re(optpls, df, scaled_data, plstraindata)

    # label generation with regression stats
    df2 = labelling(df2, limxc, limdelc, mre, 'mre')
    scaled_data2, labelset2, trset2, sc2 = splitting(df2, 'QSRR')

    # classification with new labels
    clf2, clfstats2 = classify(labelset2)
    classify_stats(clfstats2)
    labeltestset2 = labelset2[1], labelset2[3]
    classify_plot(clf2, labeltestset2)
    df2 = add_status(clf2, df2, scaled_data2, 'QSRR')
    print('')
    print('---Completion of Training and Testing Procedure--_-')
    print('')
    print('')
    print('')
    return sc, clf, clf2, optpls, mre


def validate(df_valid, models, limxc, limdelc):
    print('----------Commencing Validation Procedure----------')
    print('')

    sc, gbc, gbc2, optpls, mre = models

    df_valid = labelling(df_valid, limxc, limdelc, method='delc')
    xdata = df_valid[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                      'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    ydata = df_valid[['tR / min', 'labels']]

    xdata = sc.transform(xdata)

    print('-----------------Sequest Modelling-----------------')
    label_ydata = ydata['labels']
    y_pred_gbc1 = gbc.predict(xdata)
    classify_stats([label_ydata, y_pred_gbc1])
    classify_plot(gbc, [xdata, label_ydata])
    df_valid = add_status(gbc, df_valid, [xdata, ydata], 'Sequest')
    print('')

    print('------------------QSRR Modelling-------------------')
    tr_ydata = ydata['tR / min']
    y_pred_pls = optpls.predict(xdata)
    regress_plot([tr_ydata, y_pred_pls])
    print('')

    df_valid2 = add_re(optpls, df_valid, [xdata, ydata])
    df_valid2 = labelling(df_valid2, limxc, limdelc, mre, method='mre')

    print('-------------Sequest + QSRR Modelling--------------')
    xdata2 = df_valid2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                        'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                        're']]
    label_ydata2 = df_valid2['labels']

    y_pred_gbc2 = gbc2.predict(xdata2)
    classify_stats([label_ydata2, y_pred_gbc2])
    classify_plot(gbc2, [xdata2, label_ydata2])
    df_valid2 = add_status(gbc2, df_valid2, [xdata2, ydata], 'QSRR')
    print('')

    print('--------Completion of Validation Procedure---------')
    print('')
    print('')
    print('')
