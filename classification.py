import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix


def classify_gbc(labelset, label_params=None):
    x_train, x_test, y_train, y_test = labelset

    # GradientBoostingClassifer model
    model = GradientBoostingClassifier()
    if label_params is not None:
        clf = GradientBoostingClassifier(**label_params)
    model.fit(x_train, y_train)

    y_hat_train = model.predict(x_train)
    y_hat_test = model.predict(x_test)
    feature_importance = model.feature_importances_

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test], [feature_importance]


def classify_xgbc(labelset, label_params=None):
    x_train, x_test, y_train, y_test = labelset

    # GradientBoostingClassifer model
    model = XGBClassifier()
    if label_params is not None:
        model = XGBClassifier(**label_params)
    model.fit(x_train, y_train)

    y_hat_train = model.predict(x_train)
    y_hat_test = model.predict(x_test)
    feature_importance = model.feature_importances_

    return model, [x_train, y_train, y_hat_train], [x_test, y_test, y_hat_test], [feature_importance]


def classify_stats(data, title=None):  # title should be a string
    x_test, y_test, y_hat_test = data

    # clf model statistics
    cm = confusion_matrix(y_test, y_hat_test)
    table = pd.DataFrame(cm, columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    sens = tp / (tp + fn)  # also known as recall
    spec = tn / (tn + fp)
    mcc = ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if title is not None:
        print('----------- Classification Model Stats ------------')
        print(table)
        print('{} Model'
              'Accuracy: {:.2f}\n'
              'Sensitivity: {:.2f}\n'
              'Specificity: {:.2f}\n'
              'Matthews CC: {:.2f}'.format(title, acc, sens, spec, mcc))
        print('--------------- End of Statistics -----------------')

    return acc, sens, spec, mcc


def classify_plot(testdata):
    x_test, y_test, y_hat_test = testdata
    feature_importance = optdata

    if feature_importance is not None:
        # Features importance in clf model
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5  # to rank from most important to least
        labels = ['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
                  'error']
        sorted_labels = [labels[i] for i in sorted_idx]
        fig4, ax4 = plt.subplots()
        ax4.barh(pos, feature_importance[sorted_idx], color='C1', align='center')
        ax4.set_title('Variables Importance')
        ax4.set_ylabel('Variables')
        ax4.set_xlabel('Normalised Importance')
        ax4.set_yticks(pos)
        ax4.set_yticklabels(sorted_labels)

    # Probabilistic Distribution in clf model
    y_test_good_ind = (y_test == 1)
    y_test_bad_ind = ~y_test_good_ind
    x = np.linspace(1, len(y_hat_testproba) - 1, len(y_hat_testproba), dtype='int')

    fig5, ax5 = plt.subplots()
    ax5.scatter(x[y_test_bad_ind], y_hat_testproba[y_test_bad_ind], c='C3')
    ax5.scatter(x[y_test_good_ind], y_hat_testproba[y_test_good_ind], c='C2')
    lim1 = [np.min(ax5.get_xlim()), np.max(ax5.get_xlim())]
    lim2 = [0.5, 0.5]
    ax5.set_title('Prediction Probability Distribution')
    ax5.set_ylabel('Prediction Probability')
    ax5.plot(lim1, lim2, c='k')
