import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import radians, sin, cos, sqrt, atan2

# Load the trained model and data
@st.cache_data
def load_data(nrows):
    data = pd.read_csv('train(2).csv', nrows=nrows, low_memory=False)
    return data

@st.cache_data
def preprocess_data(data):
    data = data.dropna()
    data = data.drop(((data[data['fare_amount'] <= 0.1])).index, axis=0)
    data = data.drop(((data[data['passenger_count'] == 0])).index, axis=0)
    data = data.drop(data[data['passenger_count'] > 10].index, axis=0)
    data = data.drop(((data[data['pickup_latitude'] < -90])).index, axis=0)
    data = data.drop(((data[data['pickup_latitude'] > 90])).index, axis=0)
    data = data.drop(((data[data['pickup_longitude'] < -180])).index, axis=0)
    data = data.drop(((data[data['pickup_longitude'] > 180])).index, axis=0)
    data = data.drop(((data[data['dropoff_latitude'] < -90])).index, axis=0)
    data = data.drop(((data[data['dropoff_latitude'] > 90])).index, axis=0)
    data = data.drop(((data[data['dropoff_longitude'] < -180])).index, axis=0)
    data = data.drop(((data[data['dropoff_longitude'] > 180])).index, axis=0)
    
    data['distance'] = haversine_distance(data['pickup_latitude'], data['pickup_longitude'], data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['day_of_week'] = data['pickup_datetime'].dt.dayofweek
    data['month'] = data['pickup_datetime'].dt.month
    data['hour'] = data['pickup_datetime'].dt.hour
    
    data = data.drop(((data[data['distance'] > 1000])).index, axis=0)
    data = data.drop(((data[data['distance'] == 0])).index, axis=0)
    
    return data

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Load and preprocess the data
data = load_data(10_000_000)
data = preprocess_data(data)

# Split the data
X = data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance', 'day_of_week', 'month', 'hour']]
y = data['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=1, random_state=42)
random_forest.fit(X_train, y_train)

# App title
st.title('NYC Taxi Fare Prediction')

# User inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    pickup_longitude = st.sidebar.number_input('Pickup Longitude', min_value=-180.0, max_value=180.0, value=-73.985428)
    pickup_latitude = st.sidebar.number_input('Pickup Latitude', min_value=-90.0, max_value=90.0, value=40.748817)
    dropoff_longitude = st.sidebar.number_input('Dropoff Longitude', min_value=-180.0, max_value=180.0, value=-73.985428)
    dropoff_latitude = st.sidebar.number_input('Dropoff Latitude', min_value=-90.0, max_value=90.0, value=40.748817)
    passenger_count = st.sidebar.slider('Passenger Count', 1, 10, 1)
    day_of_week = st.sidebar.slider('Day of Week', 0, 6, 0)
    month = st.sidebar.slider('Month', 1, 12, 1)
    hour = st.sidebar.slider('Hour', 0, 23, 0)
    
    distance = haversine_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
    
    data = {'pickup_longitude': pickup_longitude,
            'pickup_latitude': pickup_latitude,
            'dropoff_longitude': dropoff_longitude,
            'dropoff_latitude': dropoff_latitude,
            'passenger_count': passenger_count,
            'distance': distance,
            'day_of_week': day_of_week,
            'month': month,
            'hour': hour}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combine user input features with the entire dataset
# This will be useful for scaling, if necessary
input_data = pd.concat([input_df, X], axis=0)

# Select only the first row (the user input data)
prediction_input = input_data[:1]

# Predict fare amount
prediction = random_forest.predict(prediction_input)

st.subheader('Prediction')
st.write(f'The predicted fare amount is: ${prediction[0]:.2f}')

# Evaluate model
y_pred = random_forest.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

st.subheader('Model Performance')
st.write(f'Root Mean Squared Error on test data: {rmse:.2f}')
