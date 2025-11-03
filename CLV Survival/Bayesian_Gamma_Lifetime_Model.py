import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data, calculate_alive_path, calibration_and_holdout_data
from lifetimes.datasets import load_cdnow_summary

from lifetimes.plotting import \
    plot_frequency_recency_matrix, \
    plot_probability_alive_matrix, \
    plot_period_transactions, \
    plot_history_alive, \
    plot_cumulative_transactions, \
    plot_calibration_purchases_vs_holdout_purchases, \
    plot_transaction_rate_heterogeneity, \
    plot_dropout_rate_heterogeneity

pd.set_option("display.max_rows",50)
pd.set_option("display.max_columns",50)

bgf_loaded = BetaGeoFitter()
bgf_loaded.load_model('bgf.pkl')

df_rft = pd.read_csv('df_ch_predict_purch_alive.csv', index_col=0)
df_rft.info()
# print(df_rft.shape)
# select customers with monetary value > 0
df_rftv = df_rft[df_rft["monetary_value_cal"] > 0]
# print(df_rftv.shape)
pd.options.display.float_format = '{:,.2f}'.format
# print(df_rftv.describe())

# Gamma-Gamma model requires a Pearson correlation close to 0
# between purchase frequency and monetary value

corr_matrix = df_rftv[["monetary_value_cal", "frequency_holdout"]].corr()
corr = corr_matrix.iloc[1,0]
print("Pearson correlation: %.3f" % corr)

# fitting the Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=1e-01)
ggf.fit(
        penalizer=0.1,
        frequency=df_rftv["frequency_holdout"],
        monetary_value=df_rftv["monetary_value_cal"],
        weights=None,
        verbose=True,
        tol=1e-06,
        q_constraint=True)
pd.options.display.float_format = '{:,.3f}'.format
print(ggf.summary)
ggf.save_model('ggf.pkl')