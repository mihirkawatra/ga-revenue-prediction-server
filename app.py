import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
import urllib.request
import pickle
import numpy as np
import os
from datetime import datetime

from utils import *


app = Flask(__name__)

# load the model from disk
MODEL_DIR = './models/'
STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
PREDICTION_FOLDER = STATIC_FOLDER + '/predictions'

print('[INFO] : Model loading ................')
model = pickle.load(open(MODEL_DIR+'gacrp_xgboost_cpu.pkl', 'rb'))
print('[INFO] : Model loaded')

@app.route('/')
def home():
    return render_template('index.html')

def predict(fullpath):
    cols = ['channelGrouping', 'visitNumber', 'device_browser',
       'device_deviceCategory', 'device_isMobile', 'device_operatingSystem',
       'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country',
       'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region',
       'geoNetwork_subContinent', 'totals_bounces', 'totals_hits',
       'totals_newVisits', 'totals_pageviews', 'trafficSource_adContent',
       'trafficSource_campaign', 'trafficSource_isTrueDirect',
       'trafficSource_keyword', 'trafficSource_medium',
       'trafficSource_referralPath', 'trafficSource_source', 'dayofweek',
       'month', 'day', 'hour', 'browser_category', 'browser_operatingSystem',
       'source_country', 'count_hits_nw_domain', 'sum_hits_nw_domain',
       'count_pvs_nw_domain', 'sum_pvs_nw_domain']
    df = load_df(fullpath)
    print('[INFO] : Data loaded from uploaded file')
    EXCLUDED_COLS = ['date', 'fullVisitorId', 'visitId', 'visitStartTime', 'totals_transactionRevenue']
    df = process_dfs(df, EXCLUDED_COLS)
    fvid = df[['fullVisitorId']].copy()
    df.drop(EXCLUDED_COLS, axis=1, inplace=True)
    df = df[cols]
    print('[INFO] : Preprocessed Dataframe')
    result = model.predict(df)
    y_test_vec = np.expm1(np.maximum(0, result))
    submission = submit(y_test_vec, fvid)
    print('[INFO] : Predictions Generated')
    filename = 'results_'+str(datetime.now().isoformat(timespec='seconds'))+'.csv'
    fullpath = os.path.join(PREDICTION_FOLDER, filename)
    submission.to_csv(fullpath, index=False)
    print('[INFO] : Saved Results File')
    return filename

# Process file and predict his label
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        print('[INFO] : Upload Received')
        file = request.files['image']
        ext = file.filename.split('.')[-1]
        if ext not in ('xlsx', 'csv'):
            abort(400, 'File Extension not supported. Please use ".xlsx" or ".csv" format.')
        else:
            fullname = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(fullname)
            print('[INFO] : File Saved. Pushing to prediction pipeline.')
            filename = predict(fullname)
            return render_template('predict.html', result_file_name=filename)

@app.route('/predict/<filename>')
def send_file(filename):
    print(os.path.exists(os.path.join(PREDICTION_FOLDER, filename)))
    try:
        return send_from_directory(PREDICTION_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    app.run(debug=True)  # auto-reload on code change
