import pandas as pd
from func import histplot

file = 'C://Users/e0031599/Desktop/iteration_metrics_20000iters_run1.csv'

df = pd.read_csv(file)

# retrieve column names
column = df.columns.values.tolist()

for i in range(df.shape[1]):
    y_data = df[[column[i]]]
    figure = histplot(y_data, column[i], ' ')
    name = 'test/Graph of {} .png'.format(column[i])
    figure.savefig(name)
    print('Generated Figure {}'.format(column[i]))
