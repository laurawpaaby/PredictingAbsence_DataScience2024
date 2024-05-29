#### HELPER FUNCTIONS PROPHET ####
# This script contains custom functions for forecasting using Prophet and plotting the results 

#### LOAD PACKAGES
# For data wrangling
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Prophet
from prophet import Prophet
from prophet.plot import plot_plotly


#### DATA PREP ####
def ts_data_prep(data):
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data_timeseries = data.filter(['Afdelinger','Stilling_niv1','Antal_Timer','Timestamp'], axis=1)

    # Group data to only have one datapoint per position per department per day
    grouped = data_timeseries.groupby(["Timestamp","Afdelinger","Stilling_niv1"])
    sum_antal_timer = grouped["Antal_Timer"].sum()
    sum_df = sum_antal_timer.to_frame().reset_index(level =["Afdelinger","Stilling_niv1"])

    dep_list = sum_df["Afdelinger"].unique()
    pos_list = sum_df["Stilling_niv1"].unique()

    return dep_list, pos_list, sum_df



#### SUBSET DATA ####
def subset_data(dep, pos, sum_df):
    
    # Create subset
    sub_df_dep = sum_df.loc[sum_df["Afdelinger"] == dep].drop("Afdelinger",axis = 1) #Subset department
    sub_dep_pos = sub_df_dep.loc[sub_df_dep["Stilling_niv1"] == pos].drop("Stilling_niv1", axis = 1) # Subset position
    sub_dep_pos = sub_dep_pos.reset_index()

    # Rename columns
    df_proph = sub_dep_pos.rename(columns={'Timestamp': 'ds',
                        'Antal_Timer': 'y'})
        
    return df_proph



#### MAKE FORECAST ####
def make_forecast(df_proph, dep, pos, periods = 30, freq = 'MS'):
    
    # Create and fit model
    model = Prophet(interval_width=0.80)
    model.fit(df_proph)
    print(f'Model for {dep}, {pos} has been fitted')

    # Save model
    from prophet.serialize import model_to_json
    with open(f'./Prophet_forecasting/model_{dep}_{pos}.json', 'w') as fout:
        fout.write(model_to_json(model))
    
    print("saved model")

    # Make forecast
    future_dates = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future_dates)

    forecast.to_csv(f'./Prophet_forecasting/forecast_{dep}_{pos}.csv')

    return model, forecast



#### PLOTTING FUNCTIONS ####
# --- Set y_lim for plots
def set_ylim(df):
    print(df['y'].max())
    if pd.isna(df['y'].max()):
        y_lim = 50
    else: y_lim = df['y'].max() + 10
    return y_lim



# --- Plot full timeseries
def plot_full_single_pos(df_merged_wide,df_proph, dep, pos):

    # Get the latest date with true data
    start_forecast = df_proph['ds'].iloc[-1]

    # Subset into two dataset to include only past and future preds
    preds_past = df_merged_wide[df_merged_wide['ds']<= start_forecast]
    preds_future = df_merged_wide[df_merged_wide['ds']>= start_forecast]
        
    # Set figure size
    plt.figure(figsize=(10, 5))

    # Plot the true number of hours of absence
    plt.scatter(df_merged_wide['ds'], df_merged_wide['y'], marker='o', label='True',color = 'sandybrown',s=10)
             
    # --- Plot predictions
    # Plot uncertainty around predictions 
    plt.fill_between(df_merged_wide['ds'],
                        df_merged_wide['yhat_lower'], 
                        df_merged_wide['yhat_upper'], 
                        color='lightblue', 
                        alpha=0.2, 
                        label='Uncertainty',
                        linewidth=1)
                        
    # Plot past predictions (yhat)
    plt.plot(preds_past['ds'], preds_past['yhat'], marker='o', label='Predicted, past',markersize=4, color = 'midnightblue',alpha = .3)

    # Plot future predictions (yhat)
    plt.plot(preds_future['ds'], preds_future['yhat'], marker='o', label='Predicted, future',markersize=4, color = 'midnightblue')

    # Plot Aesthetics 
    hfont = {'fontname':'DejaVu Serif'}
    plt.title(f'Absence: {dep} – {pos}',**hfont,horizontalalignment="center",weight='bold')
    plt.xlabel('Date',**hfont)
    plt.ylabel('Daily hours of absence',**hfont)
    plt.ylim((-5,set_ylim(df_merged_wide)))

    # Display legend
    plt.legend(loc='upper right')

    # Set theme
    sns.set_context('paper')
    sns.set_theme(context='paper',style = "whitegrid")

    # Show the plot
    plt.grid(True)
    plt.show()

    return plt



# --- Plot subset of time series
def plot_subset_single_pos(df_merged_wide, earliest_date, df_proph, dep, pos):

    subset = df_merged_wide[df_merged_wide['ds'] > earliest_date]

   # Get the latest date with true data
    start_forecast = df_proph['ds'].iloc[-1]

    # Subset into two dataset to include only past and future preds
    preds_past = subset[subset['ds']<= start_forecast]
    preds_future = subset[subset['ds']>= start_forecast]
    
    # Set figure size
    plt.figure(figsize=(10, 5))

    # Plot the true number of hours of absence
    plt.plot(subset['ds'], subset['y'], marker='o', label='True',color = 'sandybrown',markersize=2,linestyle='dashed') 
    
    # --- Plot predictions
    # Plot uncertainty around predictions 
    plt.fill_between(subset['ds'],
                    subset['yhat_lower'], 
                    subset['yhat_upper'], 
                    color='lightblue', 
                    alpha=0.2, 
                    label='Uncertainty',
                    linewidth=1)
                    
    # Plot past predictions (yhat)
    plt.plot(preds_past['ds'], preds_past['yhat'], marker='o', label='Predicted, past',markersize=4, color = 'steelblue')

    # Plot future predictions (yhat)
    plt.plot(preds_future['ds'], preds_future['yhat'], marker='o', label='Predicted, future',markersize=4, color = 'midnightblue')

    # Add titles and labels
    hfont = {'fontname':'DejaVu Serif'}
    plt.title(f'Absence: {dep} – {pos}, {earliest_date}–',**hfont,horizontalalignment="center",weight='bold')
    plt.xlabel('Date',**hfont)
    plt.ylabel('Daily hours of absence',**hfont)
    plt.ylim((-5,set_ylim(subset)))

    # Display legend
    plt.legend(loc='upper right')

    # Set theme
    sns.set_context('paper')
    sns.set_theme(context='paper',style = "whitegrid")

    # Show the plot
    plt.grid(True)
    plt.show()

    return plt



# --- Plot different positions in the same department
def plot_department(dep_forecast,start_date,end_date,dep):

    # Subset the dataset
    dep_forecast_subset = dep_forecast.loc[dep_forecast['ds'] > start_date]
    dep_forecast_subset = dep_forecast_subset.loc[dep_forecast_subset['ds'] < end_date]

    # Set figure size
    plt.figure(figsize=(10, 5))
    
    # Specify colors
    color_dict = {'Adm.personale (8M_03)': 'mediumpurple', 'Lægepersonale (8M_01)':'seagreen',
        'Plejepersonale (8M_02)':'steelblue', 'Øvr.sundh.pers (8M_04)':'sandybrown',
        'ServiceTeknisk (8M_05)':'firebrick', 'Øvr.personale (8M_08)':'grey', 'Samling':'black'
        }

    for category in dep_forecast['position'].unique():
        sub = dep_forecast_subset.loc[dep_forecast_subset['position'] == category]
        color = color_dict.get(category)

        plt.plot(sub['ds'],sub['y'], label=f'{category} - True values',color = color, alpha = .3, linewidth = 1,linestyle='dashed',marker='o',markersize=4)
        plt.plot(sub['ds'],sub['yhat'], label=f'{category} - Predicted values',color = color, alpha = .9, linewidth = 1,marker='o',markersize=4)
    
    # Add titles and labels
    hfont = {'fontname':'DejaVu Serif'}
    plt.title(f'Absence: {dep}: {start_date} – {end_date}',**hfont,horizontalalignment="center",weight='bold')
    plt.xlabel('Date',**hfont)
    plt.ylabel('Daily hours of absence',**hfont)
    plt.ylim((-5,set_ylim(dep_forecast_subset)))

    # Display legend
    plt.legend(loc = 'upper right',prop={'size': 6},bbox_to_anchor=(1.1, 1))

    # Set theme
    sns.set_context('paper')
    sns.set_theme(context='paper',style = "whitegrid")

    # Show the plot
    plt.grid(True)
    plt.show()

    return plt


#### TIME SERIES COMPONENTS ####
# --- Get components
def get_components(model_name, periods = 30, freq = 'D'):

    # Load model
    from prophet.serialize import model_from_json
    with open(f'./Prophet_forecasting/model_{model_name}.json', 'r') as fin:
        model = model_from_json(fin.read())

    # Make predictions
    future_dates = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future_dates)

    return(forecast, model)



# --- Plot components of Prophet Model
def plot_components(forecast,model):

    # Plot the components
    plt = model.plot_components(forecast)

    # Customize the colors
    axes = plt.get_axes()
    colors = ['midnightblue','steelblue','skyblue']

    # Adjust subplot parameters to make room for a title
    plt.subplots_adjust(top=0.94)

    # Add a title to the overall figure
    plt.suptitle('Components of Seasonality', size=14, family='Serif',weight = 'bold')

    # Set theme
    sns.set_context('paper')
    sns.set_theme(context='paper',style = "whitegrid")

    for ax, color in zip(axes, colors):
        for line in ax.get_lines():
            line.set_color(color)

        # Set font properties for the title and labels
        ax.xaxis.label.set_fontfamily('Serif')
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontfamily('Serif')
        ax.yaxis.label.set_fontweight('bold')

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('Serif')

    return plt



