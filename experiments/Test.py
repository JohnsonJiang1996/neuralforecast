import numpy as np
import pandas as pd
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF

# Split data and declare panel dataset

# Split data and declare panel dataset
Y_df = AirPassengersDF
Y_train_df = Y_df[Y_df.ds <= '1959-12-31']
Y_train2_df = Y_df[(Y_df.ds >= '1956-01-01') & (Y_df.ds <= '1959-12-31')]  # Train from 1951-01-01 to 1959-12-31
Y_test_df = Y_df[Y_df.ds > '1959-12-31']  # Test after 1959-12-31


# Rest of your code...

# Fit and predict with NBEATS and NHITS models

from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_and_plot(Y_train_df, title):
    # Fit and predict with NBEATS and NHITS models
    horizon = len(Y_test_df)
    models = [NBEATS(input_size=2 * horizon, h=horizon, max_steps=50),
              NHITS(input_size=2 * horizon, h=horizon, max_steps=50)]
    nf = NeuralForecast(models=models, freq='M')
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()

    # Plot predictions
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
    plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')

    plot_df[['y', 'NBEATS', 'NHITS']].plot(ax=ax, linewidth=2)

    ax.set_title(title, fontsize=22)
    ax.set_ylabel('Monthly Passengers', fontsize=20)
    ax.set_xlabel('Timestamp [t]', fontsize=20)
    ax.legend(prop={'size': 15})
    ax.grid(True)
    plt.show()

    # Calculate RMSE
    nbeats_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['NBEATS'])
    nhits_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['NHITS'])

    print(f"N-BEATS RMSE: {nbeats_RMSE}")
    print(f"N-HITS RMSE: {nhits_RMSE}")


# Call the function with different training data
train_and_plot(Y_train_df, 'AirPassengers Forecast with Y_train_df')
train_and_plot(Y_train2_df, 'AirPassengers Forecast with Y_train2_df')
