import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import time
import time as timewithsleep
import yfinance as yf # https://pypi.org/project/yfinance/
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.momentum import ROCIndicator
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
import xgboost as xgb
import plotly.graph_objects as go
from typing import TypeVar
from xgboost import XGBClassifier
import plotly.express as px

T = TypeVar('T')

###########
# sidebar #
###########
user_input = st.sidebar.text_input("Input stock", "AAPL")
today = datetime.date.today()
before = today - datetime.timedelta(days=700)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
end_time = st.sidebar.slider("End time:", value=(time(7, 00)))

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date time: `%s`' % (
        start_date, datetime.datetime.combine(end_date, end_time)))
else:
    st.sidebar.error('Error: End date must fall after start date.')

menu=["LSTM", "XGBoost"]
choices = st.sidebar.selectbox("Select Model", menu)
indicators=["Close", "ROC"]
indicator = st.sidebar.selectbox("Select Indicator", indicators)

###########
# Train model #
#############

#For LSTM MOdel ------------------------------


def create_train_test_LSTM(df, epoch, b_s, ticker_name, indicator):
    df_filtered = df.filter([indicator])
    print(df_filtered)
    print(df.filter(['Close']))
    dataset = df_filtered.values

    #Training Data
    training_data_len = math.ceil(len(dataset) * .7)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0: training_data_len, :]

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(train_data[i-60:i, 0])
        y_train_data.append(train_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    #Testing Data
    test_data = scaled_data[training_data_len - 60:, :]

    x_test_data = []
    y_test_data = dataset[training_data_len:, :]

    for j in range(60, len(test_data)):
        x_test_data.append(test_data[j - 60:j, 0])

    x_test_data = np.array(x_test_data)

    x_test_data = np.reshape(
        x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(x_train_data.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train_data, y_train_data,
              batch_size=int(b_s), epochs=int(epoch))
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write("Predicted vs Actual Results for LSTM")
    st.write("Stock Prediction on Test Data for - ", ticker_name)

    predictions = model.predict(x_test_data)
    predictions = scaler.inverse_transform(predictions)

    train = df_filtered[:training_data_len]
    valid = df_filtered[training_data_len:]
    valid['Predictions'] = predictions
    valid['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
    new_valid = valid.reset_index()
    new_valid.drop('index', inplace=True, axis=1)
    st.dataframe(new_valid)
    st.markdown('')
    st.write("Plotting Actual vs Predicted ")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(14, 8))
    plt.title('Actual vs Predicted Using LSTM Model', fontsize=20)
    plt.plot(valid[[indicator, 'Predictions']])
    plt.legend(['Actual', 'Predictions'], loc='upper left', prop={"size": 20})
    st.pyplot()



# #For XGBoost Model ------------------------------

def create_train_test_XGB(df, indicator):
    n_estimators = 100             # Number of boosted trees to fit. default = 100
    max_depth = 10                 # Maximum tree depth for base learners. default = 3
    learning_rate = 0.2            # Boosting learning rate (xgb’s “eta”). default = 0.1
    min_child_weight = 1           # Minimum sum of instance weight(hessian) needed in a child. default = 1
    subsample = 1                  # Subsample ratio of the training instance. default = 1
    colsample_bytree = 1           # Subsample ratio of columns when constructing each tree. default = 1
    colsample_bylevel = 1          # Subsample ratio of columns for each split, in each level. default = 1
    gamma = 2                      # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0

    model_seed = 1042
    
    from xgboost import XGBRegressor
    # xgb = XGBClassifier()
    xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, gamma=0.01,
                importance_type='gain', learning_rate=0.05, max_delta_step=0,
                max_depth=8, min_child_weight=1, missing=None, n_estimators=400,
                n_jobs=1, nthread=None, objective='reg:squarederror',
                random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                seed=None, silent=None, subsample=1, verbosity=1)
    # get data

    df_Stock = df.copy() #[features_selected]
    
    df_Stock['Diff'] = df_Stock['Close'] - df_Stock['Open']
    df_Stock['High-low'] = df_Stock['High'] - df_Stock['Low']
    
    st.write('Training Selected Machine Learning models for ', user_input)
    st.markdown('Your **_final_ _dataframe_ _for_ Training** ')
    st.write(df_Stock)
    my_bar = st.progress(0)
    for percent_complete in range(100):
        timewithsleep.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    # st.success('Training Completed!')

    # Create
    features = df_Stock.drop(columns=indicator, axis=1)
    features = df_Stock.drop(columns=['Date'], axis=1)
    target = df_Stock[indicator]


    data_len = df_Stock.shape[0]
    print('Historical Stock Data length is - ', str(data_len))

    #create a chronological split for train and testing
    train_split = int(data_len * 0.9)
    print('Training Set length - ', str(train_split))

    val_split = train_split + int(data_len * 0.08)
    print('Validation Set length - ', str(int(data_len * 0.1)))

    print('Test Set length - ', str(int(data_len * 0.02)))

    # Splitting features and target into train, validation and test samples 

    X_train, X_val, X_test = features[:train_split], features[train_split:val_split], features[val_split:]
    Y_train, Y_val, Y_test = target[:train_split], target[train_split:val_split], target[val_split:]

    #print shape of samples
    print(X_train.shape, X_val.shape, X_test.shape)
    print(Y_train.shape, Y_val.shape, Y_test.shape)

    #######
    xgb.fit(X_train, Y_train)

    # Y_train_pred = xgb.predict(X_train)
    Y_val_pred = xgb.predict(X_val)
    # Y_test_pred = xgb.predict(X_test)

    # # update_metrics_tracker()

    fig = plt.figure(figsize=(8,8))
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgb.feature_importances_))], xgb.feature_importances_.tolist(), tick_label=X_test.columns)
    plt.title('Feature importance of the technical indicators.')
    plt.show()
    
    # self.plot_prediction()
    print('Predicted vs Actual for ', user_input)
    st.write('Predicted vs Actual for ', user_input)
    df_pred = pd.DataFrame(Y_val.values, columns=['Actual'], index=Y_val.index)
    df_pred['Predicted'] = Y_val_pred
    df_pred['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
    df_pred = df_pred.reset_index()
    
    # df_pred.loc[:, 'Date'] = pd.to_datetime(df_pred['Date'],format='%Y-%m-%d')
    # print('Stock Prediction on Test Data - ',df_pred)
    st.write('Stock Prediction on Test Data for - ',user_input)
    st.write(df_pred)

    print('Plotting Actual vs Predicted for - ', user_input)
    st.write('Plotting Actual vs Predicted for - ', user_input)
    fig = df_pred[['Actual', 'Predicted']].plot()
    df_pred.rename(columns={'Date':'index'}).set_index('index')
    plt.title('Actual vs Predicted Stock Prices')
    #plt.show()
    #st.write(fig)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


##############
# Stock data #
##############

# Download data
df = yf.download(user_input,start= start_date,end= datetime.datetime.combine(end_date, end_time), progress=False)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.date
df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10)))*100
df = df.dropna()
st.dataframe(df)


if choices == 'LSTM':
    create_train_test_LSTM(df, 300, 1024, user_input, indicator)
elif choices == 'XGBoost':
    create_train_test_XGB(df, indicator)

df = df.rename(columns={'Date':'index'}).set_index('index')
# Price of change 
roc = ROCIndicator(df['Close']).roc()

# Bollinger Bands
indicator_bb = BollingerBands(df['Close'])
bb = df
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb = bb[['Close','bb_h','bb_l']]

# Moving Average Convergence Divergence
macd = MACD(df['Close']).macd()

# Resistence Strength Indicator
rsi = RSIIndicator(df['Close']).rsi()

###################
# Set up main app #
###################

# Plot ROC
st.write('Stock Rate Of Change')
st.line_chart(roc)

# Plot the prices and the bolinger bands
st.write('Stock Bollinger Bands')
st.line_chart(bb)

progress_bar = st.progress(0)


# Plot MACD
st.write('Stock Moving Average Convergence Divergence (MACD)')
st.area_chart(macd)

# Plot RSI
st.write('Stock RSI ')
st.line_chart(rsi)

# Data of recent days
st.write('Recent data ')
st.dataframe(df.tail(10))

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download excel file</a>' # decode b'abc' => abc

st.markdown(get_table_download_link(df), unsafe_allow_html=True)
