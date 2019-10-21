# Importing packages
import pandas as pd
import func
from preprocessing import labelling, splitting, data_restrict
from classification import classify_xgbc, classify_stats
from regression import regress_pls, regress_gbr, regress_xgbr


def traintest(modeldata, limxc, limdelc, clf_params=None, reg_params=None):
    # Label for SEQUEST
    df = labelling(modeldata, limxc, limdelc, method='delc')
    scaled_data, labelset, trset, sc, tr_max = splitting(df, 'Sequest')

    # SEQUEST Model (currently available choice: gbc/xgbc)
    # clf, clf_traindata, clf_testdata, _ = classify_gbc(labelset, clf_params)  # feature importance is muted for now
    clf, clf_traindata, clf_testdata, _ = classify_xgbc(labelset, clf_params)  # feature importance is muted for now
    acc_train1, sens_train1, spec_train1, mcc_train1 = classify_stats(clf_traindata)
    acc_test1, sens_test1, spec_test1, mcc_test1 = classify_stats(clf_testdata)

    # QSRR Model (currently available choice: pls/gbr/xgbr)
    # reg, reg_traindata, reg_testdata = regress_pls(trset, reg_params)
    # reg, reg_traindata, reg_testdata = regress_gbr(trset, reg_params)
    reg, reg_traindata, reg_testdata = regress_xgbr(trset, reg_params)

    mre_train = func.get_mre(reg_testdata[1], reg_testdata[2])
    rmsre_train = func.get_rmsre(reg_traindata[1], reg_traindata[2])
    rmsre_test = func.get_rmsre(reg_testdata[1], reg_testdata[2])

    # Label for Improved Sequest, inclusion of relative error
    df2 = func.add_error(reg, df, scaled_data)
    df2 = labelling(df2, limxc, limdelc, mre_train, method='mre')
    scaled_data2, labelset2, trset2, sc2, tr_max2 = splitting(df2, 'Improved Sequest')

    # Improved SEQUEST Model
    clf2, clf_traindata2, clf_testdata2, _ = classify_xgbc(labelset2)  # feature importance is muted for now
    acc_train2, sens_train2, spec_train2, mcc_train2 = classify_stats(clf_traindata2)
    acc_test2, sens_test2, spec_test2, mcc_test2 = classify_stats(clf_testdata2)

    # Collation of Statistics
    model = (sc, sc2, clf, clf2, reg, mre_train)
    stats = (acc_train1, sens_train1, spec_train1, mcc_train1,
             rmsre_train,
             acc_train2, sens_train2, spec_train2, mcc_train2,
             acc_test1, sens_test1, spec_test1, mcc_test1,
             rmsre_test,
             acc_test2, sens_test2, spec_test2, mcc_test2)

    return model, stats


def validate(validationdata, models, limxc, limdelc):
    sc, sc2, clf, clf2, reg, mre_train = models

    # Label for SEQUEST
    df_valid = labelling(validationdata, limxc, limdelc, method='delc')

    # Scaling of Data
    x_data = df_valid[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                       'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = df_valid[['tR / min', 'labels']]  # y_data is not scaled for now
    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns)

    '''
    # Restricting validation set
    if text is True:
        print('Initial shape of validation Set : {}'.format(df_valid.shape))

    df_valid, x_data, y_data = data_restrict([df_valid, x_data, y_data], [tr_max, reg_traindata], ad=False)

    if text is True:
        print('Final shape of validation Set : {}'.format(df_valid.shape))
    '''
    # Splitting
    x_data_clf = x_data[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp']]
    x_data_reg = x_data[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]

    y_data_clf = y_data['labels']
    y_data_reg = y_data['tR / min']

    # SEQUEST Model
    y_hat_clf1 = clf.predict(x_data_clf).ravel()
    acc1, sens1, spec1, mcc1 = classify_stats([x_data_clf, y_data_clf, y_hat_clf1])

    # QSRR Model
    y_hat_reg = reg.predict(x_data_reg).ravel()
    rmsre_valid = func.get_rmsre(y_data_reg, y_hat_reg)

    # Label for Improved Sequest
    df_valid2 = func.add_error(reg, df_valid, [x_data, y_data])
    df_valid2 = labelling(df_valid2, limxc, limdelc, mre_train, method='mre')

    # Scaling
    x_data2 = df_valid2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                         'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                         'error']]
    y_data2 = df_valid2[['tR / min', 'labels']]  # y_data is not scaled for now
    x_data2 = pd.DataFrame(sc2.transform(x_data2), columns=x_data2.columns)

    # Splitting
    x_data_clf2 = x_data2[['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                           'error']]
    y_data_clf2 = y_data2['labels']

    # Improved SEQUEST Model
    y_hat_clf2 = clf2.predict(x_data_clf2)
    acc2, sens2, spec2, mcc2 = classify_stats([x_data_clf2, y_data_clf2, y_hat_clf2])

    # Collation of Statistics
    validstats = (acc1, sens1, spec1, mcc1, rmsre_valid, acc2, sens2, spec2, mcc2)

    return validstats
