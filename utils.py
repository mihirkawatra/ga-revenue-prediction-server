import pandas as pd
import numpy as np
import json
import os

def load_df(csv_path):
    features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',
                'visitNumber', 'visitStartTime', 'device_browser',
                'device_deviceCategory', 'device_isMobile', 'device_operatingSystem',
                'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country',
                'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region',
                'geoNetwork_subContinent', 'totals_bounces', 'totals_hits',
                'totals_newVisits', 'totals_pageviews', 'totals_transactionRevenue',
                'trafficSource_adContent', 'trafficSource_campaign',
                'trafficSource_isTrueDirect', 'trafficSource_keyword',
                'trafficSource_medium', 'trafficSource_referralPath',
                'trafficSource_source']
    JSON_COLS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ext = csv_path.split('.')[-1]
    if ext == "xlsx":
        df = pd.read_excel(csv_path,
                                converters={ column: json.loads for column in JSON_COLS },
                                dtype={ 'date': str, 'fullVisitorId': str, 'sessionId': str }
                                )
    else:
        df = pd.read_csv(csv_path,
                                converters={ column: json.loads for column in JSON_COLS },
                                dtype={ 'date': str, 'fullVisitorId': str, 'sessionId': str }
                                )
    print('Load {}'.format(csv_path))
    df.reset_index(drop=True, inplace=True)
    for col in JSON_COLS:
        col_as_df = pd.json_normalize(df[col])
        col_as_df.columns = ['{}_{}'.format(col, subcol) for subcol in col_as_df.columns]
        df = df.drop(col, axis=1).merge(col_as_df, right_index=True, left_index=True)
    return df

def process_date_time(df):
    print('process date')
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    return df

def process_format(df):
    print('process format')
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        df[col] = df[col].astype(float)
    df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    return df

def process_device(df):
    print('process device')
    df['browser_category'] = df['device_browser'] + '_' + df['device_deviceCategory']
    df['browser_operatingSystem'] = df['device_browser'] + '_' + df['device_operatingSystem']
    df['source_country'] = df['trafficSource_source'] + '_' + df['geoNetwork_country']
    return df

def process_geo_network(df):
    print('process geo network')
    df['count_hits_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    df['sum_hits_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    df['count_pvs_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    df['sum_pvs_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    return df

def process_categorical_cols(df, excluded_cols):
    # Label encoding
    objt_cols = [col for col in df.columns if col not in excluded_cols and df[col].dtypes == object]
    for col in objt_cols:
        df[col], indexer = pd.factorize(df[col])

    bool_cols = [col for col in df.columns if col not in excluded_cols and df[col].dtypes == bool]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Fill NaN
    numb_cols = [col for col in df.columns if col not in excluded_cols and col not in objt_cols]
    for col in numb_cols:
        df[col] = df[col].fillna(0)

    return df

def process_dfs(df, excluded_cols):
    print('Dropping repeated columns')
    cols_to_drop = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    df.drop(cols_to_drop, axis=1, inplace=True)
    print('Extracting features')
    print('Training set:')
    df = process_date_time(df)
    df = process_format(df)
    df = process_device(df)
    df = process_geo_network(df)
    print('Postprocess')
    df = process_categorical_cols(df, excluded_cols)
    return df

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def submit(predictions, submission):
    submission.loc[:, 'PredictedLogRevenue'] = predictions
    submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0.0).apply(lambda x : 0.0 if x < 0 else x)
    grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
    grouped_test['PredictedLogRevenue'] = np.log1p(grouped_test['PredictedLogRevenue'])
    grouped_test['PredictedLogRevenue'] = grouped_test['PredictedLogRevenue'].fillna(0.0)
#     grouped_test.to_csv(filename, index=False)
    return grouped_test
