import math
import streamlit as st
import time as timewithsleep
import matplotlib.pyplot as plt
import pandas as pd

def create_train_test_XGB(df, indicator, user_input):
    from xgboost import XGBRegressor
    # xgb = XGBClassifier()
    xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, gamma=0.01,
                importance_type='gain', learning_rate=0.05, max_delta_step=0,
                max_depth=8, min_child_weight=1, missing=1, n_estimators=400,
                n_jobs=1, nthread=None, objective='reg:squarederror',
                random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                seed=None, silent=None, subsample=1, verbosity=1)
    # get data

    df_Stock = df.copy() #[features_selected]
    
    df_Stock['Diff'] = df_Stock['Close'] - df_Stock['Open']
    df_Stock['High-low'] = df_Stock['High'] - df_Stock['Low']
    
    test_dataset = df_Stock.tail(math.ceil(len(df_Stock) * .1))
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

    #create a chronological split for train and testing
    train_split = int(data_len * 0.9)

    val_split = train_split + int(data_len * 0.1)

    # Splitting features and target into train, validation and test samples 

    X_train, X_val, X_test = features[:train_split], features[train_split:val_split], features[val_split:]
    Y_train, Y_val, Y_test = target[:train_split], target[train_split:val_split], target[val_split:]

    #######
    xgb.fit(X_train, Y_train)

    Y_val_pred = xgb.predict(X_val)

    # # update_metrics_tracker()

    fig = plt.figure(figsize=(8,8))
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgb.feature_importances_))], xgb.feature_importances_.tolist(), tick_label=X_test.columns)
    plt.title('Feature importance of the technical indicators.')
    plt.show()
    
    # self.plot_prediction()
    st.write('Predicted vs Actual for ', user_input)
    df_pred = pd.DataFrame(Y_val.values, columns=['Actual'], index=Y_val.index)
    df_pred['Predicted'] = Y_val_pred
    df_pred['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
    df_pred = df_pred.reset_index()
    
    st.write('Stock Prediction on Test Data for - ',user_input)
    st.write(df_pred)

    st.write('Plotting Actual vs Predicted for - ', user_input)
    fig = df_pred[['Actual', 'Predicted']].plot()
    df_pred.rename(columns={'Date':'index'}).set_index('index')
    plt.title('Actual vs Predicted Stock Prices')
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # predict next day
    # Predict price
    predicted_stock_price = xgb.predict(X_val)

    # Convert to price
    st.write('Stock price next day', predicted_stock_price[len(predicted_stock_price)-1])
