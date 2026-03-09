import pandas as pd

# Load your dataset
df = pd.read_csv("invocations_per_function_md.anon.d01.csv")

# Identify the time-series columns (they are numbered 1 to 1440)
time_cols = [str(i) for i in range(1, 1441)]

# Group by application (HashApp) and aggregate the time series
# You can use sum() or mean() depending on what the values represent
app_level = df.groupby("HashApp")[time_cols].sum().reset_index()


# save the csv
app_level.to_csv("app_level.csv", index=False)

# Now each row = one application
# Each column = activity at that minute
print(app_level.head())

# If you want to extract a single application's time series:
app_id = app_level.iloc[0]["HashApp"]
ts = app_level[app_level["HashApp"] == app_id][time_cols].values.flatten()


print(f"Time series for app {app_id}:")
print(ts)
