import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data, calculate_alive_path #, calibration_yand_holdout_data,

from lifetimes.plotting import \
    plot_frequency_recency_matrix, \
    plot_probability_alive_matrix, \
    plot_period_transactions, \
    plot_history_alive, \
    plot_cumulative_transactions, \
    plot_calibration_purchases_vs_holdout_purchases, \
    plot_transaction_rate_heterogeneity, \
    plot_dropout_rate_heterogeneity

sns.set(rc={'image.cmap': 'coolwarm'})

pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.0f}'.format

df1 = pd.read_csv('raw_ml.csv', index_col=0, dtype={'kunnr': np.str, 'matnr': np.str})
# df1["date"] = pd.to_datetime(df1["date"], format='%Y-%m-%d').dt.date
# df1 = df1.astype({'date': 'datetime64[ns]'})
df1["date"] = pd.to_datetime(df1["date"]).dt.date
print(df1)
df1.info()
print(df1.iloc[0, 2])
print(type(df1.iloc[0, 2]))

# recency, frequency, T
# dfx = df1[df1["kunnr"] == df1.iloc[0, 2]]
dfx = df1[df1['kunnr'] == '0050107913']
print(dfx)
xmax_date = dfx["date"].max()
xmin_date = dfx["date"].min()

# recency:
print("customer minimum date:", xmin_date)
print("customer maximum date:", xmax_date)
xrec = (xmax_date - xmin_date).days
print("recency:", xrec)  # recency = time span between first and last purchase

# age T:
xmaxall_date = df1["date"].max()
print("population maximum date:", xmaxall_date)
xage = (xmaxall_date - xmin_date).days # age T
print("T:", xage)

# frequency:
xfreq = len(dfx.groupby("date"))-1    # frequency: periods with repeat purchases
print("frequency:", xfreq)