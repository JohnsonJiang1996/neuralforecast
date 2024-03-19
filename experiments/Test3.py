import numpy as np
import pandas as pd
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, MSTL, Theta, AutoRegressive, AutoETS, AutoTheta
from statsforecast.utils import AirPassengersDF, Reversed_AirPassengersDF, AirPassengersDF2, AirPassengersDF3

# Split data and declare panel dataset

# Split data and declare panel dataset

Y_df = AirPassengersDF3
Y_train_df = Y_df[Y_df.ds <= '1959-12-31']
# Y_train2_df = Y_df[(Y_df.ds >= '1956-01-01') & (Y_df.ds <= '1959-12-31')]  # Train from 1951-01-01 to 1959-12-31
Y_test_df = Y_df[Y_df.ds > '1959-12-31']  # Test after 1959-12-31

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

    # Fit and predict with all models
    sf = StatsForecast(
        models=[
            AutoARIMA(season_length=12),
            MSTL(season_length=12),
            Theta(season_length=12),
            AutoRegressive(lags=12),
            AutoETS(season_length=12),
            AutoTheta( season_length=12)
        ],
        freq='M'
    )
    sf.fit(Y_train_df)
    sf_predictions = sf.predict(h=horizon, level=[95])
    sf_predictions = sf_predictions.reset_index()
    # print(sf_predictions.columns)
    # sf_predictions.rename(columns={'mean': 'StatsForecast'}, inplace=True)

    # Merge predictions
    model_columns = ['unique_id', 'ds', 'AutoARIMA', 'MSTL', 'Theta', 'AutoRegressive', 'AutoETS', 'AutoTheta']
    Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
    # 合并所有模型的预测结果到 Y_hat_df DataFrame
    Y_hat_df = Y_hat_df.merge(sf_predictions[model_columns], how='left', on=['unique_id', 'ds'])

    # Plot predictions
    fig, ax = plt.subplots(1, 1, figsize=(26, 10))
    plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')

    # 使用所有模型的预测结果列进行绘图
    plot_df[['y', 'NBEATS', 'NHITS', 'AutoARIMA', 'MSTL', 'Theta', 'AutoRegressive', 'AutoETS', 'AutoTheta']].plot(
        ax=ax, linewidth=2)

    for column in ['y']:
        # Extract the data for the specific column
        data = plot_df[column]

        # Plot the data in red color if the date is between 1949-01-01 and 1949-12-31
        data.loc['1949-01-01':'1951-12-31'].plot(ax=ax, linewidth=2, linestyle='dashed', color='red', label='Generated_y')

    ax.set_title(title, fontsize=22)
    ax.set_ylabel('Monthly Passengers', fontsize=20)
    ax.set_xlabel('Timestamp [t]', fontsize=20)
    ax.legend(prop={'size': 15})
    ax.grid(True)

    # Calculate RMSE for each model
    nbeats_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['NBEATS'])
    nhits_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['NHITS'])
    autoarima_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['AutoARIMA'])
    mstl_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['MSTL'])
    theta_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['Theta'])
    autoregressive_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['AutoRegressive'])
    autoets_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['AutoETS'])
    autotheta_RMSE = calculate_rmse(Y_test_df['y'], Y_hat_df['AutoTheta'])

    # Add RMSE to the plot
    ax.text(-0.1, 0.8,
            f"N-BEATS RMSE: {nbeats_RMSE:.2f}\nN-HITS RMSE: {nhits_RMSE:.2f}\nAutoARIMA RMSE: {autoarima_RMSE:.2f}\nMSTL RMSE: {mstl_RMSE:.2f}\nTheta RMSE: {theta_RMSE:.2f}\nAutoRegressive RMSE: {autoregressive_RMSE:.2f}\nAutoETS RMSE: {autoets_RMSE:.2f}\nAutoTheta RMSE: {autotheta_RMSE:.2f}",
            transform=ax.transAxes, fontsize=12, horizontalalignment='center', verticalalignment='bottom',
            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5))

    plt.show()

# Call the function with different training data
train_and_plot(Y_train_df, 'AirPassengers Forecast with Original Training Data')
# train_and_plot(Y_train2_df, 'AirPassengers Forecast with Modified Training Data')