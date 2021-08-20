import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

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
    test_dataset = df_filtered.tail(math.ceil(len(dataset) * .3))

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


    # predict next day
    dataset_test = test_dataset[indicator][len(test_dataset)-60:len(test_dataset)].to_numpy()
    dataset_test = np.array(dataset_test)

    inputs = dataset_test
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    
    X_test = []
    no_of_sample = len(dataset_test)

    # Get last data
    X_test.append(inputs[no_of_sample - 60:no_of_sample, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict price
    predicted_stock_price = model.predict(X_test)

    # Convert to price
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    # Add date
    dataset_test = np.append(dataset_test, predicted_stock_price[0], axis=0)
    inputs = dataset_test
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    # print('Stock price next day', predicted_stock_price[0][0])
    st.write('Stock ', indicator, ' next day value:', predicted_stock_price[0][0])

