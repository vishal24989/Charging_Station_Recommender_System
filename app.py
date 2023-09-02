import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import geopy.distance

# Load data
@st.cache
def load_data():
    soc_df = pd.read_csv('SOC_Prediction_Dataset_Bangalore_Updated.csv')
    charging_station_df = pd.read_csv('Charging_Station_Recommender_System_Dataset_Bangalore_Updated.csv')
    return soc_df, charging_station_df

soc_df, charging_station_df = load_data()

# Prepare the model for SOC prediction
def prepare_model(df):
    X = df.drop(['Timestamp', 'Current Vehicle Location', 'SOC (%)'], axis=1)
    y = df['SOC (%)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write('Model RMSE: ', rmse)
    return model

model = prepare_model(soc_df)

# SOC Prediction
st.header('State of Charge (SOC) Prediction')
current = st.number_input('Current (A)')
voltage = st.number_input('Voltage (V)')
temperature = st.number_input('Temperature (°C)')
battery_capacity = st.number_input('Battery Capacity (Ah)')
accessory_load = st.number_input('Accessory Load (W)')
elevation = st.number_input('Elevation (m)')
temperature_outside = st.number_input('Temperature Outside (°C)')

if st.button('Predict SOC'):
    input_features = np.array([current, voltage, temperature, battery_capacity, accessory_load, elevation, temperature_outside]).reshape(1, -1)
    prediction = model.predict(input_features)
    st.write('Predicted SOC: ', prediction[0])

# Charging Station Recommender System
st.header('Charging Station Recommender System')
current_location = st.text_input('Current Location (latitude, longitude)')

if st.button('Recommend Charging Stations'):
    current_location = tuple(map(float, current_location.split(',')))
    distances = [geopy.distance.geodesic(current_location, loc).km for loc in charging_station_df['Charging_Station_Location']]
    charging_station_df['Distance'] = distances
    recommended_stations = charging_station_df.nsmallest(5, 'Distance')
    st.write(recommended_stations)
