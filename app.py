import numpy as np
import pandas as pd
import matplotlib as plt
import pandas_datareader as data
from yahoo_fin import stock_info as si
from keras.models import load_model
import streamlit as st





# start = '2015-01-01'
# end = '2024-03-07'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

start=st.date_input('Enter Start Date', value="default_value_today", min_value=None, max_value=None, key=None, help=None, on_change=None, args=None, kwargs=None, format="YYYY/MM/DD", disabled=False, label_visibility="visible")

end = st.date_input('Enter End Date', value="default_value_today", min_value=None, max_value=None, key=None, help=None, on_change=None, args=None, kwargs=None, format="YYYY/MM/DD", disabled=False, label_visibility="visible")

df = si.get_data(user_input, start_date=start, end_date=end)

#Describing Data
st.subheader('Data from 2015 to 2024')
st.write(df.describe())


#visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100ma (100 day's Moving Average)")
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100ma & 200ma")
# st.subheader("(100 & 200 day's Moving Average)")
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100)
# plt.plot(ma200)
plt.plot(ma100, label='100 day Moving Average')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.plot(ma200, label = '200 day Moving Average')
plt.plot(df.close)
st.pyplot(fig)


#splitting data into traning and testing

data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)]) 
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#spliting data into xtrain and ytrain


#load my model
model = load_model('keras_model.keras')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#final Graph


st.subheader("Prediction vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label='Original Price')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
