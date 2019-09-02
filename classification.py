import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix


# input from preprocessing


# classification
def classify(traintestset):
    x_train, x_test, y_train, y_test = traintestset

    # GBC model
    clf = GradientBoostingClassifier()

    clf.fit(x_train, y_train)
    y_testpred = clf.predict(x_test)

    return [clf, [y_test, y_testpred]]


def classify_plot(clf, testset):
    x_test, y_test = testset

    # Features importance in clf model
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5  # to rank from most important to least
    labels = ['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
              'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
              'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
              're']
    sorted_labels = [labels[i] for i in sorted_idx]
    fig4, ax4 = plt.subplots()
    ax4.barh(pos, feature_importance[sorted_idx], align='center')
    ax4.set_xlabel('Relative Importance')
    ax4.set_yticks(pos)
    ax4.set_yticklabels(sorted_labels)

    # Probabilistic Distribution in clf model
    y_test_good_ind = (y_test == 1)
    y_test_bad_ind = ~y_test_good_ind
    y = clf.predict_proba(x_test)[:, 1]
    x = np.linspace(1, len(y) - 1, len(y), dtype='int')

    fig5, ax5 = plt.subplots()
    ax5.scatter(x[y_test_bad_ind], y[y_test_bad_ind], c='r', edgecolors='k')
    ax5.scatter(x[y_test_good_ind], y[y_test_good_ind], c='b', edgecolors='k')
    lim1 = [np.min(ax5.get_xlim()), np.max(ax5.get_xlim())]
    lim2 = [0.5, 0.5]
    ax5.plot(lim1, lim2, c='k')

    plt.show()


def classify_stats(stats):
    y_test, y_testpred = stats

    # clf model statistics
    cm = confusion_matrix(y_test, y_testpred)
    table = pd.DataFrame(cm, columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    sens = tp / (tp + fn)  # also known as recall
    prec = tp / (tp + fp)
    spec = tn / (tn + fp)
    mcc = ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    # plotting performance stats
    print('--------------------Model Stats--------------------')
    print(table)
    print('Accuracy: {:.2f}'.format(acc))
    print('Sensitivity: {:.2f}'.format(sens))
    print('Specificity: {:.2f}'.format(spec))
    print('Matthews CC: {:.2f}'.format(mcc))
    print('-----------End of Performance Statistics-----------')


def add_status(gbc, data, scaleddata, name):
    x_data = scaleddata[0]
    y_data = scaleddata[1]['labels'].values
    y_pred = gbc.predict(x_data)

    label = 'status_{}'.format(name)
    status = []

    for i in range(len(y_data)):
        if y_data[i] == 1 and y_pred[i] == 1:
            status.append('tp')
        elif y_data[i] == 0 and y_pred[i] == 1:
            status.append('fp')
        elif y_data[i] == 0 and y_pred[i] == 0:
            status.append('tn')
        elif y_data[i] == 1 and y_pred[i] == 0:
            status.append('fn')

    data[label] = status

    return data
