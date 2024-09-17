import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv("data\\dataset\\dataset.csv")

# Convert 'date' column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Set 'date' as the index for time series analysis
df.set_index("date", inplace=True)

# Sort the index by date
df = df.sort_index()

# Define the target variable (Venta_piezas) and exogenous variables
target = df["Venta_piezas"]
exog = df[
    [
        "Promocion",
        "EventoEspecial_Venta_piezas",
        "holiday",
        "porcentaje_gasto_alimentos",
    ]
]

# Train-test split (80% train, 20% test)
train_size = int(len(target) * 0.8)
train_target, test_target = target[:train_size], target[train_size:]
train_exog, test_exog = exog[:train_size], exog[train_size:]

# Fit SARIMAX model with weekly seasonality (52 weeks)
model = SARIMAX(
    train_target, exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)
)
results = model.fit()

# Forecast for the test set
forecast_steps = len(test_target)
forecast = results.get_forecast(steps=forecast_steps, exog=test_exog)
forecast_df = forecast.summary_frame()

# Extract forecasted values and confidence intervals
forecast_values = forecast_df["mean"]
ci_lower = forecast_df["mean_ci_lower"]
ci_upper = forecast_df["mean_ci_upper"]

# Evaluate the forecast using MAE, RMSE, and MAPE
mae = mean_absolute_error(test_target, forecast_values)
rmse = np.sqrt(mean_squared_error(test_target, forecast_values))
mape = np.mean(np.abs((test_target - forecast_values) / test_target)) * 100

# Output the evaluation results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Extend exogenous variables for 52 weeks (use zeros, averages, or realistic future estimates)
future_exog = pd.DataFrame(
    {
        "Promocion": [0] * 52,  # Example: assume no promotion in future
        "EventoEspecial_Venta_piezas": [0] * 52,  # Example: no special events
        "holiday": [0] * 52,  # Example: assume no holidays
        "porcentaje_gasto_alimentos": [47]
        * 52,  # Use average value or last known percentage
    },
    index=pd.date_range(
        start=df.index[-1] + pd.Timedelta(weeks=1), periods=52, freq="W"
    ),
)

# Forecast for the next 52 weeks
future_forecast = results.get_forecast(steps=52, exog=future_exog)
future_forecast_df = future_forecast.summary_frame()

# Combine the forecasted values with the actual test values for comparison
forecast_output = pd.DataFrame(
    {
        "Actual": test_target,
        "Forecast": forecast_values,
        "CI Lower": ci_lower,
        "CI Upper": ci_upper,
    }
)

# Export the forecast data to CSV
forecast_output.to_csv("data\\dataset\\forecast_results.csv", index=True)

# Export the future 52-week forecast to CSV
future_forecast_df["Sucursal"] = "Norte 286"
future_forecast_df["Cadena"] = "SA"
future_forecast_df["Recurso"] = "68524 Alimentos"
future_forecast_df.to_csv("data\\dataset\\future_forecast_52_weeks.csv", index=True)

print("Done.")
