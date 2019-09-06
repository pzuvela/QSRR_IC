import pandas as pd
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from scipy.optimize import minimize, Bounds


# input from preprocessing


# classification
def classify(traintestset, n_splits, optimise=False):
    x_train, x_test, y_train, y_test = traintestset

    # GBC model
    clf = GradientBoostingClassifier()

    if optimise is True:
        print(' ')
        print('    - Commencing Optimisation of Classification Model -')

        def fun(x):
            # Descaling Parameters
            n_est = int(np.round(np.exp(x[0]), decimals=0))
            min_sam = int(np.round(np.exp(x[1]), decimals=0))
            lr = x[2] ** 2
            max_depth = int(np.round(np.exp(x[3]), decimals=0))

            opt_clf = GradientBoostingClassifier(n_estimators=n_est,
                                                 min_samples_split=min_sam,
                                                 learning_rate=lr,
                                                 max_depth=max_depth)

            # K-Fold object
            kfold = KFold(n_splits=n_splits)
            # Scoring object
            scorer = make_scorer(roc_auc_score)
            # CV score
            score = cross_val_score(opt_clf, x_train, y_train, cv=kfold, scoring=scorer)

            return -np.mean(score)

        # Creating bounds
        n_est_min, n_est_max = 100, 1000
        min_sam_min, min_sam_max = 5, 50
        lr_min, lr_max = 0.01, 0.1
        max_depth_min, max_depth_max = 1, 5
        bounds = Bounds([np.log(n_est_min), np.log(min_sam_min), np.sqrt(lr_min), np.log(max_depth_min)],
                        [np.log(n_est_max), np.log(min_sam_max), np.sqrt(lr_max), np.log(max_depth_max)])

        # Pre-loading initial values
        n_est0 = np.log(rand.uniform(n_est_min, n_est_max))
        min_sam0 = np.log(rand.uniform(min_sam_min, min_sam_max))
        lr0 = np.sqrt(rand.uniform(lr_min, lr_max))
        max_depth0 = np.log(rand.uniform(max_depth_min, max_depth_max))
        initial = np.array([n_est0, min_sam0, lr0, max_depth0])
        print('    ---------------- Initial Parameters ---------------')
        print('    n_estimators: {:.0f}\n'
              '    min_sample_split: {:.0f}\n'
              '    learning_rate: {:.2f} \n'
              '    max_depth: {:.0f}'
              .format(np.exp(n_est0), np.exp(min_sam0), np.square(lr0), np.exp(max_depth0))
              )
        print('    ---------------------------------------------------')

        # Begin Optimisation
        opt = minimize(fun, initial, method='trust-constr', bounds=bounds, options={'maxiter': 100})
        x_dict = {'n_estimators': int(np.round(np.exp(opt.x[0]), decimals=0)),
                  'min_samples_split': int(np.round(np.exp(opt.x[1]), decimals=0)),
                  'learning_rate': np.square(opt.x[2]),
                  'max_depth': int(np.round(np.exp(opt.x[3]), decimals=0))
                  }

        # Implementing optimised parameters
        clf.set_params(**x_dict)

        print('    ----------------- Final Parameters ----------------')
        print('    n_estimators: {:.0f}\n'
              '    min_sample_split: {:.0f}\n'
              '    learning_rate: {:.2f} \n'
              '    max_depth: {:.0f}\n'
              '    Final CV-ROC: {:.2f}'
              .format(np.exp(opt.x[0]), np.exp(opt.x[1]), np.square(opt.x[2]), np.exp(opt.x[3]), -opt.fun)
              )
        print('    ---------------------------------------------------')
        print('    ------------ Completion of Optimisation -----------')
        print(' ')

    clf.fit(x_train, y_train)
    y_testpred = clf.predict(x_test)

    return [clf, [y_test, y_testpred]]


"""
        # K-Fold object
        kf = KFold(n_splits=n_splits)

        # Parameter grid
        param_grid = {'learning_rate': np.linspace(0.001, 0.1, step),
                      'n_estimators': np.linspace(10, 1000, step, dtype='int'),
                      'min_samples_split': np.linspace(5, 50, step, dtype='int'),
                      'max_depth': np.linspace(1, 10, step, dtype='int')
                      }
        clf_cv = GridSearchCV(clf, cv=kf, param_grid=param_grid)

        clf_cv.fit(x_train, y_train)

        clf.set_params(**clf_cv.best_params_)

        clf.fit(x_train, y_train)

        # clf.fit(x_train, y_train)
        y_testpred = clf.predict(x_test)

"""


def classify_plot(clf, testset):
    x_test, y_test = testset

    # Features importance in clf model
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5  # to rank from most important to least
    labels = ['MH+', 'Charge', 'm/z', 'XC', 'Delta Cn', 'Sp',
              'error']
    sorted_labels = [labels[i] for i in sorted_idx]
    fig4, ax4 = plt.subplots()
    ax4.barh(pos, feature_importance[sorted_idx], align='center')
    ax4.set_title('Variables Importance')
    ax4.set_ylabel('Variables')
    ax4.set_xlabel('Normalised Importance')
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
    ax5.set_title('Prediction Probability Distribution')
    ax5.set_ylabel('Prediction Probability')
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
    print('----------- Classification Model Stats ------------')
    print(table)
    print('Accuracy: {:.2f}\n'
          'Sensitivity: {:.2f}\n'
          'Specificity: {:.2f}\n'
          'Matthews CC: {:.2f}'.format(acc, sens, spec, mcc))
    print('---------- End of Performance Statistics ----------')


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
