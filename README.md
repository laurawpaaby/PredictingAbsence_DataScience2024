# Comparing Modeling Techniques for Predicting Absence: A Case Study from Regionshospital Gødstrup 🏥 ♥️ 🤖 
The present repository contains the code develped for an exam paper in the course Data Science, Predicting, and Forecasting at the Master's in Cognitive Science, Aarhus University by Klara Fomsgaard and Laura Paaby

#### Data availability
Due to privacy restrictions, the analyzed data is not included in the current repository. Access may be granted upon request, with joined consent from Gødstrup Sygehus and the authors.

## Setup
**Step 1** Run ```setup.sh```

To replicate the setup, we have included a bash script that automatically 

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Runs the script
5. Deactivates the virtual environment

## Usage

### Regression Models and Feature Importance 
**Step 2** Run ```data_prep_1.ipynb```, ```data_prep_2.ipynb``` and ```descriptive_plots_and_data_split.ipynb``` <br>
Running these notebooks will: 
- Preprocess and clean data 
- Generate additional features
- Scale independent variables 
- Split the data into train (80%) and test (20%) subsets
- Visualize the raw data 

**Step 3** Run ```regressors_GRID.py``` <br>
This script conducts a comprehensive grid search across all regressors, identifying and storing the optimal parameters that yield the highest performance in RegMod_Performance.

**Step 4** Run ```fitting_best_params.py``` <br>
This script fits all models using their optimal parameters determined previously. The performance metrics ($R^2$, $MAE$, $RMSE$) for these models are evaluated on the test dataset and recorded.

**Step 5** Run ```baselinemodel.py``` <br>
This script creates two baselinemodels:
- A model which always predicts the mean of the target
- A model which always predicts the a value corresponding to the previous datapoint

**Step 6** Run ```feature_imp.py``` <br>
This script calculates the permutation feature importance and their standard deviations for the two top-performing models, XGBoost and Random Forest, and stores the results.

**Step 7** Run ```plot_script.R```<br>
This R script generates visualizations of the feature importances and the models’ predictions in comparison to the actual data values. The visualizations are stored in ./plots.

### Time Series Prediction and Forecasting

**Step 1** Run ```forecasting_subset.py``` <br>
This script fits a Prophet forecasting model for selected groups in the emergency department:
- Medical staff
- Nursing staff
- Administrative staff
The script generates plots both for the entire timeseries and a subset including data and predictions from 2024-, and stores them in 'forecasting_plots'.

## Repository Overview
```
.
├── data_prep/                                  <--- folder containing scripts related to data prep and data visualization
│   ├── data_prep_1.ipynb
│   ├── data_prep_2.ipynb
│   └── descriptive_plots_and_data_split.ipynb
│
├── plots/                                      <--- folder containing plots from feature importance analysis
├── Reg_Model_Performance/                      <--- folder with results from model comparison and feature importance
│   └── BestParameters/                         <--- folder containing the best parameters
│
├── time_series_prophet/                        <--- folder containing timeseries analysis and forecasting using Prophet
│   ├── forecasting_plots/
│   ├── create_plot_grids.py
│   ├── forecast_subset.py
│   └── helper_functions_forecasting.py
│
├── .gitignore
├── README.md
├── baselinemodels.py
├── feature_imp.py                                      
├── fitting_best_params.py   
├── plot_script.R
├── regressors_GRID.py
├── requirements.txt            
└── setup.sh
```
