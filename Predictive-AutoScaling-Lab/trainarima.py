import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the aggregated application-level dataset
# ---------------------------------------------------------
app_level = pd.read_csv("app_level.csv")

# Identify the time-series columns (1 to 1440)
time_cols = [str(i) for i in range(1, 1441)]

# ---------------------------------------------------------
# 2. Choose an application to model
#    (Here we pick the one with the highest total activity)
# ---------------------------------------------------------
app_level["total_activity"] = app_level[time_cols].sum(axis=1)
target_app = app_level.sort_values("total_activity", ascending=False).iloc[0]

app_id = target_app["HashApp"]
ts = target_app[time_cols].values.astype(float)

print(f"Training ARIMA on application: {app_id}")

# ---------------------------------------------------------
# 3. Fit ARIMA model
# ---------------------------------------------------------
# Basic baseline ARIMA(1,0,1) — simple but effective
model = ARIMA(ts, order=(1, 0, 1))
model_fit = model.fit()

print(model_fit.summary())

# ---------------------------------------------------------
# 4. Forecast the next N steps
# ---------------------------------------------------------
forecast_steps = 60  # predict the next 60 minutes
forecast = model_fit.forecast(steps=forecast_steps)

# ---------------------------------------------------------
# 5. Plot actual vs forecast
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(ts, label="Actual")
plt.plot(range(len(ts), len(ts) + forecast_steps), forecast, label="Forecast")
plt.title(f"ARIMA Forecast for App {app_id}")
plt.xlabel("Time (minutes)")
plt.ylabel("Activity")
plt.legend()
plt.tight_layout()
plt.show()
