import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import radians, sin, cos, sqrt, atan2

def load_data(nrows):
    data = pd.read_csv('train(2).csv', nrows=nrows, low_memory=False)
    return data

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

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return model, rmse

def predict_fare(model, input_data):
    return model.predict(input_data)

if __name__ == "__main__":
    data = load_data(10_000_000)
    data = preprocess_data(data)
    X = data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance', 'day_of_week', 'month', 'hour']]
    y = data['fare_amount']
    model, rmse = train_model(X, y)
    print(f"Model trained. RMSE on test data: {rmse:.2f}")
    # Save the model if needed
    import joblib
    joblib.dump(model, 'random_forest_model.pkl')
