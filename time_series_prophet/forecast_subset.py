### FORECAST ON SELECTED GROUPS ###
# This script generates forecasts for three selected groups within the emergency department
# 1. Medical personnel
# 2. Caretaking personnel
# 3. Administrative personnel

#### LOAD PACKAGES ####
# For data wrangling
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Prophet
from prophet import Prophet
from prophet.plot import plot_plotly

# Custom functions
from helper_functions_forecasting import ts_data_prep, subset_data, make_forecast
from helper_functions_forecasting import plot_full_single_pos, plot_subset_single_pos


#### LOAD DATA ####
data = pd.read_csv("./data_w_patients.csv",sep=",")
print("Loaded data")

#### PREPARE DATA ####
# Clean data
dep_list, pos_list, sum_df = ts_data_prep(data) # Function returns list of positions and departments, which can be used to run all combinations

#### RUN FORECAST AND CROSSVALIDATION FOR EACH POSITION ####
# Create dataframe for storing results
dep_forecast = pd.DataFrame(columns=['position', 'ds','yhat', 'yhat_lower','yhat_upper','y'])
metrics_forecast = pd.DataFrame(columns=['position','rmse','mae','r2'])

# Specify department and list of positions
dep = 'akutafdelingen'
pos_list = ['Plejepersonale (8M_02)','Lægepersonale (8M_01)','Adm.personale (8M_03)']

for pos in pos_list:
        # Subset data
        df_proph = subset_data(dep = dep,pos = pos, sum_df=sum_df)

        model, forecast = make_forecast(df_proph,dep,pos,periods=30,freq='D')
        print(f'model and forecast saved for {dep}, {pos}')
        
        # -- Plot
        # Create subset for plotting
        forecast_subset = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        df_merged_wide = forecast_subset.merge(df_proph,how="left")
        df_merged_wide['position'] = pos

        # Save plots
        plt_full = plot_full_single_pos(df_merged_wide,df_proph,dep,pos)
        plt_full.savefig(f'./time_series_prophet/forecasting_plots/full_{dep}_{pos}.png')

        plt_subset = plot_subset_single_pos(df_merged_wide,'2024-01-01',df_proph,dep,pos)
        plt_subset.savefig(f'./time_series_prophet/forecasting_plots/subset_{dep}_{pos}.png')
        
        # Append subset to df
        dep_forecast = pd.concat([dep_forecast,df_merged_wide])

        # -- Model evaluation

        # Baseline model
        window_size = 7
        moving_average = df_proph['y'].rolling(window=window_size).mean().iloc[-1]
        print(moving_average)



