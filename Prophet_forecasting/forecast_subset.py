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
from prophet.plot import plot_plotly, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics

# Custom functions
from helper_functions_forecasting import ts_data_prep, subset_data, make_forecast
from helper_functions_forecasting import plot_full_single_pos, plot_subset_single_pos, plot_department

# Performance metrics
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

#### LOAD DATA ####
data = pd.read_csv("/work/Exam/data_w_patients.csv",sep=",")
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
        plt_full.savefig(f'/work/Exam/prophet/selected_groups/full_{dep}_{pos}.png')

        plt_subset = plot_subset_single_pos(df_merged_wide,'2024-01-01',df_proph,dep,pos)
        plt_subset.savefig(f'/work/Exam/prophet/selected_groups/subset_{dep}_{pos}.png')
        
        # Append subset to df
        dep_forecast = pd.concat([dep_forecast,df_merged_wide])

        # -- Model evaluation

        # Baseline model
        window_size = 7
        moving_average = df_proph['y'].rolling(window=window_size).mean().iloc[-1]
        print(moving_average)

        # Calculate RMSE
        df_merged_wide.dropna(inplace=True)

        rmse = root_mean_squared_error(df_merged_wide['y'], df_merged_wide['yhat'])
        mae = mean_absolute_error(df_merged_wide['y'], df_merged_wide['yhat'])
        r2 = r2_score(df_merged_wide['y'], df_merged_wide['yhat'])

        metrics_df = pd.DataFrame([{'position':pos,'rmse':rmse,'mae':mae,'r2':r2}])

        metrics_forecast =  pd.concat([metrics_forecast,metrics_df])

        # Crossvalidation
        df_cv = cross_validation(model, initial='365 days', period='180 days', horizon = '30 days')
        df_cv_performance = performance_metrics(df_cv)
        df_cv_performance.to_csv(f'/work/Exam/prophet/selected_groups/performance_cv_table_{dep}_{pos}.csv',index = False)

        fig_cv = plot_cross_validation_metric(df_cv, metric='rmse')
        fig_cv.savefig(f'/work/Exam/prophet/selected_groups/performance_cv_rmse_{dep}_{pos}.png')



        


    
# Make plot per department
start_date='2022-08-01'
end_date='2023-02-01'
plt_department = plot_department(dep_forecast,start_date,end_date,dep)
plt_department.savefig(f'/work/Exam/prophet/selected_groups/plot_forecast_selected_{dep}.png')

# Save performance metrics
metrics_forecast.to_csv(f'/work/Exam/prophet/selected_groups/performance_metrics_forecast.csv', index = False)
