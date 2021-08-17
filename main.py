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
import lstm
import xgb
T = TypeVar('T')

###########
# sidebar #
###########
user_input = st.sidebar.text_input("Input stock", "AAPL")
today = datetime.date.today()
before = today - datetime.timedelta(days=700)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (
        start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

menu=["LSTM", "XGBoost"]
choices = st.sidebar.selectbox("Select Model", menu)
indicators=["Close", "ROC"]
indicator = st.sidebar.selectbox("Select Indicator", indicators)


##############
# Stock data #
##############

# Download data
df = yf.download(user_input, start = start_date, end= end_date, progress=False)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.date
df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10)))*100
df = df.dropna()

if choices == 'LSTM':
    lstm.create_train_test_LSTM(df, 300, 1024, user_input, indicator)
elif choices == 'XGBoost':
    xgb.create_train_test_XGB(df, indicator, user_input)

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
