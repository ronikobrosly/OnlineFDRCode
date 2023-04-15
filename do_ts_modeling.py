import json
from os.path import expanduser

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet


######################################
## READ IN WINDOWS
######################################

# read file
with open(expanduser("~/Desktop/OnlineFDRCode/nab/labels/combined_windows.json"), 'r') as myfile:
    labels = myfile.read()

labels_dict = json.loads(labels)





######################################
## ambient_temperature_system_failure
######################################

# Read in dataset
df = pd.read_csv("~/Desktop/OnlineFDRCode/nab/data/realKnownCause/ambient_temperature_system_failure.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])

initial_plot_df = df.set_index("ds")
initial_plot_df.plot(linestyle='-', lw=0.5, color = {"y": "steelblue"})
plt.show()

# Fit the model
m = Prophet(interval_width = 0.95, mcmc_samples = 0)
m.fit(df)

# Extend dataset by 5-minute units (the units of the original dataset)
future = m.make_future_dataframe(periods=1, freq = 'H')
forecast = m.predict(future)

# Plot batch forecast
fig1 = m.plot(forecast)

# Plot real, known anomalies

anomaly_windows = labels_dict["realKnownCause/ambient_temperature_system_failure.csv"]
for window in anomaly_windows:
    plt.fill_betweenx(100, pd.to_datetime(window[0]), pd.to_datetime(window[1]), color='red', alpha=.5)

plt.show()








######################################
## cpu_utilization_asg_misconfiguration
######################################

# Read in dataset
df = pd.read_csv("~/Desktop/OnlineFDRCode/nab/data/realKnownCause/cpu_utilization_asg_misconfiguration.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])

# Fit the model
m = Prophet()
m.fit(df)

# Extend dataset by 5-minute units (the units of the original dataset)
future = m.make_future_dataframe(periods=200, freq = '5T')
forecast = m.predict(future)

# Python
fig1 = m.plot(forecast)
plt.show()