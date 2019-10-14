# Importing packages
import pandas as pd
from func import get_mre
from preprocessing import labelling, splitting, data_restrict
from classification import classify, classify_stats, classify_plot
from regression import regress_pls, regress_plot, regress_gbr, regress_stats
from func import add_error


def traintest(modeldata, limxc, limdelc, n_splits, max_component, disp=False):
    if disp is True:
        print('### Commencing Training and Testing Procedure ####\n\n')

    # label generation with input limits
    df = labelling(modeldata, limxc, limdelc, method='delc')
    scaled_data, labelset, trset, sc, tr_max = splitting(df, 'Sequest')

    # classification model
    if disp is True:
        print('############### Sequest Modelling ################')
    clf, clf_traindata, clf_testdata, clf_optdata = classify(labelset, n_splits, optimise=False)

    acc_train1, sens_train1, spec_train1, mcc_train1, acc_test1, sens_test1, spec_test1, mcc_test1 \
        = classify_stats(clf_traindata, clf_testdata, text=disp)
    if disp is True:
        classify_plot(clf_testdata, clf_optdata)
        print('############ End of Sequest Modelling ############\n')

    # regression model (currently choice between pls/gbr)
    if disp is True:
        print('################## QSRR Modelling ################')
    # reg, reg_traindata, reg_testdata, reg_optdata = regress_pls(trset, n_splits, max_component)
    reg, reg_traindata, reg_testdata, reg_optdata = regress_gbr(trset, n_splits, optimise=False)

    df2 = add_error(reg, df, scaled_data)
    mre = get_mre(reg_testdata[1], reg_testdata[2])
    rmsre_train, rmsre_test = regress_stats(reg_traindata, reg_testdata, text=disp)
    if disp is True:
        regress_plot(reg_testdata, reg_traindata, reg_optdata)
        print('############# End of QSRR Modelling ##############\n\n')

    # label generation with regression stats
    df2 = labelling(df2, limxc, limdelc, mre, method='mre')

    # combination of above two model
    if disp is True:
        print('############ Sequest + QSRR Modelling ############')
    scaled_data2, labelset2, trset2, sc2, tr_max2 = splitting(df2, 'QSRR')

    # classification with new labels
    clf2, clf_traindata2, clf_testdata2, clf_optdata2 = classify(labelset2, n_splits)
    acc_train2, sens_train2, spec_train2, mcc_train2, acc_test2, sens_test2, spec_test2, mcc_test2 \
        = classify_stats(clf_traindata2, clf_testdata2, text=disp)
    if disp is True:
        classify_plot(clf_testdata2, clf_optdata2)
        print('######### End of Sequest + QSRR Modelling #########\n\n')
        print('## Completion of Training and Testing Procedure ##\n\n\n')

    # collating model for validation
    modeldata = (sc, sc2, clf, clf2, reg, mre, tr_max, reg_traindata)
    stats = (acc_train1, sens_train1, spec_train1, mcc_train1,
             rmsre_train,
             acc_train2, sens_train2, spec_train2, mcc_train2,
             acc_test1, sens_test1, spec_test1, mcc_test1,
             rmsre_test,
             acc_test2, sens_test2, spec_test2, mcc_test2)

    return modeldata, stats


def validate(validationdata, models, limxc, limdelc, disp=False):
    if disp is True:
        print('######## Commencing Validation Procedure #########\n\n')

    sc, sc2, clf, clf2, reg, mre, tr_max, reg_traindata = models

    df_valid = labelling(validationdata, limxc, limdelc, method='delc')

    x_data = df_valid[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                       'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = df_valid[['tR / min', 'labels']]

    # Scaling
    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns)

    '''
    # Restricting validation set
    if text is True:
        print('Initial shape of validation Set : {}'.format(df_valid.shape))

    df_valid, x_data, y_data = data_restrict([df_valid, x_data, y_data], [tr_max, reg_traindata], ad=False)

    if text is True:
        print('Final shape of validation Set : {}'.format(df_valid.shape))
    '''
    # Splitting x and y data for clf and reg models
    x_data_clf = x_data[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp']]
    x_data_reg = x_data[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]

    y_data_clf = y_data['labels']
    y_data_reg = y_data['tR / min']

    if disp is True:
        print('############### Sequest Modelling ################')
    y_hat_clf1 = clf.predict(x_data_clf)
    y_hat_proba1 = clf.predict_proba(x_data_clf)[:, 1]

    acc1, sens1, spec1, mcc1 = classify_stats(None, [x_data_clf, y_data_clf, y_hat_clf1, y_hat_proba1], text=disp)
    if disp is True:
        classify_plot([x_data_clf, y_data_clf, y_hat_clf1, y_hat_proba1], [None])
        print('############ End of Sequest Modelling ############\n\n')

        print('################## QSRR Modelling ################')
    y_hat_reg = reg.predict(x_data_reg).ravel()
    df_valid2 = add_error(reg, df_valid, [x_data, y_data])

    rmsre_test = regress_stats(None, [x_data_reg, y_data_reg, y_hat_reg], text=disp)
    if disp is True:
        regress_plot([x_data_reg, y_data_reg, y_hat_reg], reg_traindata)
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

    if disp is True:
        print('############ Sequest + QSRR Modelling ############')
    y_hat_clf2 = clf2.predict(x_data_clf2)
    y_hat_proba2 = clf2.predict_proba(x_data_clf2)[:, 1]

    acc2, sens2, spec2, mcc2 = classify_stats(None, [x_data_clf2, y_data_clf2, y_hat_clf2, y_hat_proba2], text=disp)
    if disp is True:
        classify_plot([x_data_clf2, y_data_clf2, y_hat_clf2, y_hat_proba2], [None])
        print('######### End of Sequest + QSRR Modelling #########\n\n')

        print('####### Completion of Validation Procedure ########\n\n\n')

    validstats = (acc1, sens1, spec1, mcc1, rmsre_test, acc2, sens2, spec2, mcc2)

    return validstats
