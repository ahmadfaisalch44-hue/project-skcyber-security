import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)
num_samples = 1000

irradiance = np.random.uniform(0, 1000, num_samples)
temperature = np.random.uniform(10, 40, num_samples)
cloud_cover = np.random.uniform(0, 100, num_samples)
humidity = np.random.uniform(20, 90, num_samples)
wind_speed = np.random.uniform(0, 15, num_samples)

efficiency_factor = 0.20
power_output = (irradiance * efficiency_factor) * (1 - cloud_cover/200) * (1 - (temperature-25)*0.005)

power_output += np.random.normal(0, 5, num_samples)
power_output = np.maximum(power_output, 0)


data = pd.DataFrame({
    'Irradiance_W/m2': irradiance,
    'Temperature_C': temperature,
    'Cloud_Cover_%': cloud_cover,
    'Humidity_%': humidity,
    'Wind_Speed_m/s': wind_speed,
    'Power_Output_kW': power_output
})

print("Dataset Preview:")
print(data.head(), "\n")


X = data.drop(columns=['Power_Output_kW'])
y = data['Power_Output_kW']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}\n")

model = RandomForestRegressor(n_estimators=100, random_state=42)


model.fit(X_train, y_train)
print("Model training complete.\n")

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} kW (Average prediction error)")
print(f"R-squared (Accuracy):      {r2 * 100:.2f}%\n")

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("--- Feature Importance ---")
print(feature_importance_df.to_string(index=False), "\n")

new_weather_forecast = pd.DataFrame({
    'Irradiance_W/m2': [850],
    'Temperature_C': [32],
    'Cloud_Cover_%': [10],
    'Humidity_%': [45],
    'Wind_Speed_m/s': [3]
})

predicted_power = model.predict(new_weather_forecast)
print("--- Prediction for Tomorrow ---")
print(f"Expected Solar Power Output: {predicted_power[0]:.2f} kW")
