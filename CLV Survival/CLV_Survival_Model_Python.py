import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data, calculate_alive_path, calibration_and_holdout_data

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

pd.set_option("display.max_rows",50)
pd.set_option("display.max_columns",50)
pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.0f}'.format

df1 = pd.read_csv('raw_ml.csv', index_col=0, dtype={'kunnr': np.str, 'matnr': np.str})
df1["date"] = pd.to_datetime(df1["date"]).dt.date
# df1.info()


# train/test split (calibration/holdout)
t_holdout = 240                                         # days to reserve for holdout period

max_date = df1["date"].max()                     # end date of observations
print("end of observations:", max_date)

max_cal_date = max_date - timedelta(days=t_holdout)     # end date of chosen calibration period
print("end of calibration period:", max_cal_date)

df_ch = calibration_and_holdout_data(
        transactions = df1,
        customer_id_col = "kunnr",
        datetime_col = "date",
        monetary_value_col = "revenues",
        calibration_period_end = max_cal_date,
        observation_period_end = max_date,
        freq = "D")

# print("customer behavior in calibration and holdout periods")
# pd.options.display.float_format = '{:,.0f}'.format
# print(df_ch)
# print(df_ch.describe())

# training: frequency
pd.options.display.float_format = '{:,.3f}'.format
x = df_ch["frequency_cal"].value_counts(normalize=True)
x = x.nlargest(15)
# print("frequency:")
# print(x.sort_index(ascending=True))

# training: axis length
max_freq = df_ch["frequency_cal"].quantile(0.98)
max_rec = df_ch["recency_cal"].max()
max_T = df_ch["T_cal"].max()

# # training
# fig = plt.figure(figsize=(8, 6))
# ax = sns.distplot(df_ch["frequency_cal"])
# ax.set_xlim(0, max_freq)
# ax.set_title("frequency (days): distribution of the customers");
# # plt.show(sns)
#
# # training
# fig = plt.figure(figsize=(8, 6))
# ax = sns.distplot(df_ch["recency_cal"])
# ax.set_xlim(0, max_rec)
# ax.set_title("recency (days): distribution of the customers")
#
#
# # training
# fig = plt.figure(figsize=(8, 6))
# ax = sns.distplot(df_ch["T_cal"])
# ax.set_xlim(0, max_T)
# ax.set_title("customer age T (days): distribution of the customers")
# plt.show()

# training: BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=1e-06)
bgf.fit(
        frequency = df_ch["frequency_cal"],
        recency = df_ch["recency_cal"],
        T = df_ch["T_cal"],
        weights = None,
        verbose = True,
        tol = 1e-06)

pd.options.display.float_format = '{:,.3f}'.format
print(bgf.summary)
bgf.save_model('bgf.pkl')

# training: does the model reflect the actual data closely enough?

# frequency of repeat transactions: predicted vs actual
fig = plt.figure(figsize=(12, 12))
plot_period_transactions(bgf);

# testing: predicted vs actual purchases in holdout period
fig = plt.figure(figsize=(7, 7))
plot_calibration_purchases_vs_holdout_purchases(bgf, df_ch)
plt.show()