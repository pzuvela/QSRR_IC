from numpy import sqrt, square, argmax, sort, vstack, transpose
from pandas import DataFrame, concat, read_csv
from glob import glob


def rmse_scorer(y_data, y_hat):
    return sqrt(square(y_hat - y_data).mean())


def fileprint(string, directory):
    with open(directory, 'a') as f:
        print(string, file=f)


def knee(x, y):
    del1, del2 = [], []
    for i in range(0, len(x) - 1):
        del1.append(y[i + 1] - y[i])

    for j in range(0, len(del1) - 1):
        del2.append(del1[j + 1] - del1[j])

    max_del2 = argmax(del2)
    kneept = x[max_del2 + 2]
    return int(kneept)


def add_true_mean_std(y_true, df):
    stats = []
    for i in range(len(df.columns)):
        label = df.columns[i]
        col_values = sort(df[label].values)
        mean = col_values.mean()
        j = int(len(col_values) * 2.5 / 100)
        lower_value = col_values[j]

        try:
            k = int(len(col_values) - j)
            upper_value = col_values[k]
        except IndexError:  # in cases where j = 0 and index exceeds
            k = int(len(col_values) - j) - 1
            upper_value = col_values[k]

        lower_limit = mean - lower_value
        upper_limit = upper_value - mean
        stats.append([mean, lower_limit, upper_limit, lower_value, upper_value])
    
    if y_true is not None:
        stats = vstack((y_true, transpose(stats)))
        indices = ['actual', 'mean', 'lower_limit', 'upper_limit', 'lower_value', 'upper_value']
    else:
        stats = transpose(stats)
        indices = ['mean', 'lower_limit', 'upper_limit', 'lower_value', 'upper_value']
        
    df_stats = DataFrame(stats, index=indices, columns=df.columns)
    df = concat([df_stats, df])
    return df


def add_mean_std(df):
    mean = df.mean(axis=0).ravel()
    std = df.std(axis=0).ravel()
    stats = [mean, std]
    df_stats = DataFrame(stats, index=['mean', 'std'], columns=df.columns)
    df = concat([df_stats, df])
    return df


def get_limits(file):
    df = read_csv(file)
    stats_list = []

    for i in range(1, len(df.columns)):
        label = df.columns[i]
        col_values = sort(df[label].values)
        j = int(len(col_values) * 2.5 / 100)
        k = int(len(col_values) - j)
        mean = col_values.mean()
        lower_value = col_values[j]
        upper_value = col_values[k]
        lower_limit = mean - lower_value
        upper_limit = upper_value - mean
        stats_list.append([label, mean, lower_limit, upper_limit, lower_value, upper_value])
        towrite = '{} limits are {:.2f}(+{:.2f};-{:.2f})'.format(label, mean, upper_limit, lower_limit)
        print(towrite)
    DataFrame(stats_list, columns=['label', 'mean', 'lower_limit', 'upper_limit', 'lower_value', 'upper_value'])\
        .to_csv('{}_stats.csv'.format(file[:-4]), index=False)


# Function to merge files from parallel runs
def merge_files(file_str, tr_exp=None):

    file_df = DataFrame()

    for file in glob(file_str):
        print('Processing', file, '...')
        file_df = file_df.append(read_csv(file, index_col=0), sort=False, ignore_index=True)

    df_merged = add_true_mean_std(tr_exp, file_df) if tr_exp is not None else add_true_mean_std(None, file_df)

    return df_merged


"""
Deprecated functions & lines:

def histplot(y_data, title, x_axis):
    fig, ax = plt.subplots()
    ax.hist(y_data, bins=20, density=True)
    x_min, x_max = ax.get_xlim()
    y_mean, y_std = norm.fit(y_data)
    p = norm.pdf(linspace(x_min, x_max, 100), y_mean, y_std)
    ax.plot(linspace(x_min, x_max, 100), p)
    ax.set_title(title)
    ax.set_ylabel('Probability Density')
    ax.set_xlabel(x_axis)
    ax.text(0.2, 0.9, s='Mean:{:.2f} +/- {:.2f}'.format(y_mean, 3*y_std), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.5))
    return fig
    
file_df = file_df[~file_df['Unnamed: 0'].str.contains('actual|mean|lower_limit|upper_limit|lower_value|'
                                                  'upper_value')].reset_index(drop=True)

"""
