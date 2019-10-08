import time
import pandas as pd
from regression import regress_gbr, regress_plot, regress_stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from multiprocessing import Pool
from func import add_true_mean_std

plot = False
fileroot = 'C://Users/e0031599/Desktop/sys'
# for sys in [1, 2, 3]:
sys = 1
# for mean in ['nomean', 'mean']:
mean = 'mean'
max_iter = 10
proc_i = 2


def model_parallel(arg_iter):
    rawdata = pd.read_csv('{}{}{}.csv'.format(fileroot, sys, mean))
    x_data = rawdata[['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']]
    y_data = rawdata[['tR / min']].values.ravel()
    x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
    n_splits = 3

    sc = StandardScaler()
    x_train = pd.DataFrame(sc.fit_transform(x_train_unscaled), columns=x_train_unscaled.columns).values
    x_test = pd.DataFrame(sc.transform(x_test_unscaled), columns=x_test_unscaled.columns).values
    x_data = pd.DataFrame(sc.transform(x_data), columns=x_data.columns).values
    trset = [x_train, x_test, y_train, y_test]

    reg, reg_traindata, reg_testdata, reg_optdata = regress_gbr(trset, n_splits, optimise=False)
    rmsre_train, rmsre_test = regress_stats(reg_traindata, reg_testdata)
    y_hat = reg.predict(x_data)

    if plot is True:
        notneeded, notneeded2, response, ax2 = regress_plot(reg_testdata, reg_traindata, reg_optdata)
        ax2.text(0.8, 0.12, 'Training %RMSE = {:.2f}'.format(rmsre_train), horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.text(0.8, 0.02, 'Testing %RMSE = {:.2f}'.format(rmsre_test), horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        response.savefig('sideQSRR/sys{}{}}.png'.format(sys, mean))
        plt.close('all')

    print('Iteration #{}/{} completed'.format(arg_iter[0] + 1, max_iter))
    return rmsre_train, rmsre_test, y_data, y_hat


if __name__ == '__main__':
    print('Initiating with {} iterations'.format(max_iter))
    start_time = time.time()

    p = Pool(processes=proc_i)
    models_final = p.map(model_parallel, zip(range(max_iter)))

    masterStats = [models_final[i][:2] for i in range(max_iter)]
    y_pred = [models_final[i][3] for i in range(max_iter)]
    y_true = models_final[0][2]

    run_time = time.time() - start_time
    print('Simulation Completed')
    print('Duration: {}\n'.format(time.strftime("%H:%M:%S", time.gmtime(run_time))))

    column = ['rmsre_train', 'rmsre_test']
    add_true_mean_std(None, pd.DataFrame(masterStats, columns=column)).to_csv(
        'sideQSRR/iteration_metrics_sys{}{}_{}iters.csv'.format(sys, mean, max_iter), header=True)

    y_pred = pd.DataFrame(y_pred)
    add_true_mean_std(y_true, y_pred).to_csv('sideQSRR/qsrr_trprediction_sys{}{}_{}iters.csv'
                                             .format(sys, mean, max_iter), header=True)