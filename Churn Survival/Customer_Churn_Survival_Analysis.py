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

bgf_loaded = BetaGeoFitter()
bgf_loaded.load_model('bgf.pkl')

raw = pd.read_csv('raw_ml.csv', index_col=0)
df_ch = pd.read_csv('df_ch_predict_purch_alive.csv', index_col=0)
df_ch.info()

df = df_ch[df_ch['frequency_cal'] > 0]
df['prob_alive'] = bgf_loaded.conditional_probability_alive(df['frequency_cal'], df['recency_cal'], df['T_cal'])
# sns.distplot(df['prob_alive']);
# plt.show()

# df['churn'] = ['churned' if p < .01 else 'not churned' for p in df['prob_alive']]
# sns.countplot(df['churn']);
# plt.show()

# sns.distplot(df[df['churn'] =='not churned']['prob_alive']).set_title('Probability alive, not churned');
# plt.show()

# ax = sns.distplot(df['predict_purch_30']).set_xlim(0, 8)
# plt.show()

custID = 'H987010530'
df_C = df.loc[custID,:]
print(df_C)

raw_C = raw[raw['kunnr'] == custID]
print(raw_C)
