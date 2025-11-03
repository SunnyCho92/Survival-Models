import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import warnings
warnings.filterwarnings(action='ignore')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

np.random.seed(1234)
_ = torch.manual_seed(123)

def gradeToCat(x):
    if x == 'K6': return 0
    elif x == 'K5': return 1
    elif x == 'K4': return 2
    elif x == 'K3': return 3
    elif x == 'K2': return 4
    elif x == 'K1': return 5
    elif x == 'P1': return 6
    elif x == 'P2': return 7
    elif x == 'P3': return 8
    elif x == 'P4': return 9
    elif x == 'P5': return 10
    elif x == 'P6': return 11
    elif x == 'M1': return 12
    elif x == 'M2': return 13
    elif x == 'M3': return 14
    elif x == 'H1': return 15
    elif x == 'H2': return 16
    elif x == 'H3': return 17
    elif x == 'A': return 18
    else: return -1

def subToCat(x):
    if x == 'AM': return 0
    elif x == 'AMA': return 1
    elif x == 'AMB': return 2
    elif x == 'DAM': return 3
    elif x == 'DM': return 4
    elif x == 'HDM': return 5
    elif x == 'KRT': return 6
    elif x == 'PDK': return 7
    elif x == 'PDM': return 8
    elif x == 'SKV': return 9
    elif x == 'SSE': return 10
    elif x == 'STE': return 11
    elif x == 'STK': return 12
    elif x == 'STM': return 13
    else: return -1

df_train = pd.read_csv('./생존모형을_위한_데이터셋.csv', index_col=0).reset_index()
df_train = df_train.dropna()
df_train.drop(['brdt', 'kunnr'], axis=1, inplace=True)

df_train['grade'] = df_train['grade'].apply(gradeToCat)
df_train['sub'] = df_train['sub'].apply(subToCat)

df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['payment']
cols_leave = ['sub', 'grade']
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

get_target = lambda df: (df['duration'].values, df['status'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val

in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

model = CoxPH(net, tt.optim.Adam)
batch_size = 256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()
plt.show()
model.optimizer.set_lr(0.01)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)

_ = log.plot()
plt.show()

print(model.partial_log_likelihood(*val).mean())

_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)

surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
plt.show()

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print(ev.concordance_td())
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
_ = ev.brier_score(time_grid).plot()
plt.show()

print(ev.integrated_brier_score(time_grid))
print(ev.integrated_nbll(time_grid))



