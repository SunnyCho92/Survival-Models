from conn_db import *
import pandas as pd
pd.set_option('mode.chained_assignment',  None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

def kunnr_pro(x):
    try: return str(int(x))
    except: return x

def matnr_channel(x):
    if x == 'RDD' or x == 'RD' or x == 'RTD' or x == 'RT':
        return '솔루니'
    elif x == 'DT' or x == 'DTD':
        return '눈높이'

def matnr_way(x):
    if x == 'RDD' or x == 'RTD' or x == 'DTD':
        return '단독'
    elif x == 'RD' or x == 'RT' or x == 'DT':
        return '병행'

def exDay(x):
    return x.days

def ex_duration(data, cc='전체'):
    if cc != '전체': data = data[data['channel_way'] == cc]
    return data['dur1'].max()

def preprocess(data, cc='전체'):
    data = data[['kunnr', 'matnr', 'zstd_stdt_01', 'zstd_eddt_01']]
    data['kunnr'] = data['kunnr'].apply(kunnr_pro)
    data['channel'] = data['matnr'].apply(matnr_channel)
    data['way'] = data['matnr'].apply(matnr_way)
    data['channel_way'] = data['channel'] + "_" + data['way']
    data['cohort'] = data['zstd_stdt_01'].apply(str).str.slice(0, 7)
    data['zstd_stdt_01'] = data['zstd_stdt_01'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    data['zstd_eddt_01'] = data['zstd_eddt_01'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    data['dur1'] = (data['zstd_eddt_01'] - data['zstd_stdt_01']).apply(exDay)
    out = data[data['zstd_eddt_01'].notnull()] # 퇴회
    out['dur1_month'] = out['dur1'] // 30

    if cc != '전체':
        data = data[data['channel_way'] == cc]
        out = out[out['channel_way'] == cc]

    return data, out, data['dur1'].max()

def retention(sql, cc='전체'):
    df = load_data(sql, 'dm')
    data, out, date_max = preprocess(df, cc)
    length = int((date_max // 30) + 1)
    g0 = data.groupby(['cohort'])['kunnr', 'matnr'].size().reset_index()
    g1 = out.groupby(['cohort', 'dur1_month'])['kunnr', 'matnr'].size().reset_index()
    g = pd.merge(g1, g0, on=['cohort'], how='right').reset_index().drop(['index'], axis=1).fillna(0)
    g.columns = ['cohort', 'dur1_month', 'km', 'total']
    g = g.astype({'dur1_month': 'int'})
    grouped = g.groupby(["cohort"])

    ls_p = []
    ls_n = []
    for name, group in grouped:
        group['km_cum'] = pd.Series(group['km'].cumsum())
        group['retention'] = group['total'] - group['km_cum']
        group['retention_percentage'] = round(group['retention'] / group['total'] * 100, 1)
        group = group.reset_index()
        p = [0]*length
        n = [0]*length
        for i in range(len(group)):
          id = group.loc[i, 'dur1_month']
          p[id] = group['retention_percentage'][i]
          n[id] = group['retention'][i]
        ls_p.append(p)
        ls_n.append(n)

    table_p = pd.DataFrame(ls_p).fillna('')
    table_p.insert(0, 'cohort', g0['cohort'])
    table_p.insert(1, 'total', g0[0])
    table_n = pd.DataFrame(ls_n).fillna('')
    table_n.insert(0, 'cohort', g0['cohort'])
    table_n.insert(1, 'total', g0[0])

    return table_p, table_n
