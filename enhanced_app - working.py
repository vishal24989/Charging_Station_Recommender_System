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
    soc_df = pd.read_csv('TripA07_with_Combined_Location.csv')
    charging_station_df = pd.read_csv('Charging_Station_Recommender_System_Dataset_Bangalore_Updated.csv')
    return soc_df, charging_station_df

soc_df, charging_station_df = load_data()

# Prepare the model for SOC prediction
def prepare_model(df):
    # List of columns to be removed
    columns_to_remove = ['Time [s]', 'min. SoC [%]', 'max. SoC [%)', 'Regenerative Braking Signal ', 
                         'Heating Power CAN [kW]', 'Requested Heating Power [W]', 
                         'max. Battery Temperature [°C]', 'displayed SoC [%]', 'Heater Signal', 
                         'Requested Coolant Temperature [°C]', 'Location', 'Cumulative Distance (km)',
                         'Cabin Temperature Sensor [°C]', 'Motor Torque [Nm]', 'Elevation [m]', 
                         'Ambient Temperature [°C]']

    # Drop the specified columns
    df = df.drop(columns=columns_to_remove, errors='ignore')

    # Separate the target variable
    X = df.drop('SoC [%]', axis=1)
    y = df['SoC [%]']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestRegressor
    model = RandomForestRegressor(random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model

model = prepare_model(soc_df)

# SOC Prediction
 # Input fields for the features
velocity = st.number_input("Velocity [km/h]", min_value=0.0, max_value=200.0, value=0.0)
throttle = st.number_input("Throttle [%]", min_value=0.0, max_value=100.0, value=0.0)
acceleration = st.number_input("Longitudinal Acceleration [m/s^2]", min_value=0.0, max_value=10.0, value=0.0)
battery_voltage = st.number_input("Battery Voltage [V]", min_value=0.0, max_value=500.0, value=0.0)
battery_current = st.number_input("Battery Current [A]", min_value=-100.0, max_value=100.0, value=0.0)
battery_temperature = st.number_input("Battery Temperature [°C]", min_value=-50.0, max_value=100.0, value=0.0)
aircon_power = st.number_input("AirCon Power [kW]", min_value=0.0, max_value=10.0, value=0.0)
heat_exchanger_temp = st.number_input("Heat Exchanger Temperature [°C]", min_value=-50.0, max_value=100.0, value=0.0)


if st.button('Predict SOC'):
    features = pd.DataFrame([[velocity, throttle, acceleration, battery_voltage, battery_current, battery_temperature, aircon_power, heat_exchanger_temp]],
                           columns=['Velocity [km/h]', 'Throttle [%]', 'Longitudinal Acceleration [m/s^2]', 
                                    'Battery Voltage [V]', 'Battery Current [A]', 'Battery Temperature [°C]', 
                                    'AirCon Power [kW]', 'Heat Exchanger Temperature [°C]'])
    prediction = model.predict(features)
    st.success(f"The predicted State of Charge (SoC) is: {prediction[0]:.2f}%")

# # Charging Station Recommender System
# st.header('Charging Station Recommender System')
# current_location = st.text_input('Current Location (latitude, longitude)')

# if st.button('Recommend Charging Stations'):
#     current_location = tuple(map(float, current_location.split(',')))
#     distances = [geopy.distance.geodesic(current_location, eval(loc)).km for loc in charging_station_df['Charging_Station_Location']]
#     charging_station_df['Distance'] = distances
    
#     # Calculate score
#     charging_station_df['Score'] = 0.25*(1-charging_station_df['Distance']) + 0.25*(1-charging_station_df['Cost_per_kWh (₹)']) + 0.25*charging_station_df['Rating'] - 0.25*charging_station_df['Queue']
    
#     recommended_stations = charging_station_df.nlargest(3, 'Score')
#     st.write(recommended_stations[['Charging_Station_Location', 'Cost_per_kWh (₹)', 'Rating', 'Queue', 'Distance']])
    
#     # Show on Google Map
#     

#     markers = f'color:blue|label:C|{current_location[0]},{current_location[1]}'
#     for i, loc in enumerate(recommended_stations['Charging_Station_Location']):
#         location = eval(loc)
#         markers += f'&markers=color:red|label:{i+1}|{location[0]},{location[1]}'
#     map_url = f'https://maps.googleapis.com/maps/api/staticmap?center={current_location[0]},{current_location[1]}&zoom=13&size=600x400&maptype=roadmap&{markers}&key={api_key}'
#     st.image(map_url)


# Charging Station Recommender System with User Preferences
st.header('Enhanced Charging Station Recommender System')

# Assuming we have a user profile for preference weights. In a real-world scenario, 
# this could be fetched from a user's profile or learned from past interactions.
user_preferences = {
    'Distance': st.slider('Preference for Distance (Higher means you prefer closer stations)', 0.0, 1.0, 0.25),
    'Cost': st.slider('Preference for Cost (Higher means you prefer cheaper stations)', 0.0, 1.0, 0.25),
    'Rating': st.slider('Preference for Rating (Higher means you prefer higher rated stations)', 0.0, 1.0, 0.25),
    'Queue': st.slider('Preference for Queue (Higher means you prefer less crowded stations)', 0.0, 1.0, 0.25),
}

current_location_preference = st.text_input('Current Location (latitude, longitude) for Enhanced Recommendation')

if st.button('Recommend Charging Stations with Preferences'):
    current_location_preference = tuple(map(float, current_location_preference.split(',')))
    distances_preference = [geopy.distance.geodesic(current_location_preference, eval(loc)).km for loc in charging_station_df['Charging_Station_Location']]
    charging_station_df['Distance'] = distances_preference
    
    # Calculate score with user preferences. This score is a weighted sum based on the user's preferences.
    charging_station_df['Preference_Score'] = (
        user_preferences['Distance']*(1-charging_station_df['Distance']) + 
        user_preferences['Cost']*(1-charging_station_df['Cost_per_kWh (₹)']) + 
        user_preferences['Rating']*charging_station_df['Rating'] - 
        user_preferences['Queue']*charging_station_df['Queue']
    )
    
    recommended_stations_preference = charging_station_df.nlargest(3, 'Preference_Score')

    # Extract coordinates for mapping
    map_data = pd.DataFrame({
        'latitude': [lat for lat, _ in recommended_stations_preference['Charging_Station_Location'].apply(lambda x: eval(x))] + [current_location_preference[0]],
        'longitude': [lon for _, lon in recommended_stations_preference['Charging_Station_Location'].apply(lambda x: eval(x))] + [current_location_preference[1]],
        'Place': ['Charging Station']*3 + ['Current Location']
    })

    # Display the map with points
    st.map(map_data)

    st.write(recommended_stations_preference[['Charging_Station_Location', 'Cost_per_kWh (₹)', 'Rating', 'Queue', 'Distance']])
    


#    # Show on Google Map for enhanced recommendation
#     markers_preference = f'color:blue|label:C|{current_location_preference[0]},{current_location_preference[1]}'
#     for i, loc in enumerate(recommended_stations_preference['Charging_Station_Location']):
#         location = eval(loc)
#         markers_preference += f'&markers=color:red|label:{i+1}|{location[0]},{location[1]}'
#     map_url_preference = f'https://maps.googleapis.com/maps/api/staticmap?center={current_location_preference[0]},{current_location_preference[1]}&zoom=13&size=600x400&maptype=roadmap&{markers_preference}&key={api_key}'
#     st.image(map_url_preference) 
