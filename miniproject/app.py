import streamlit as st
import pandas as pd
import joblib
import numpy as np
from prediction import haversine_distance, load_data, preprocess_data,train_model

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# App title
st.title(' Taxi Fare Prediction')

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

# Predict fare amount
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(f'The predicted fare amount is: ${prediction[0]:.2f}')

# Evaluate model (optional if you want to show it on the app)
#data = load_data(10_000_000)
#data = preprocess_data(data)
#X = data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance', 'day_of_week', 'month', 'hour']]
#y = data['fare_amount']
#model, rmse = train_model(X, y)

#st.subheader('Model Performance')
#st.write(f'Root Mean Squared Error on test data: {rmse:.2f}')
