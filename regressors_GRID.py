
# Making a script for all the regressors in the predict_fravaer.ipynb
# This script will be used to compare the performance of all the regressors

# IMPORT PACKAGES   
import pandas as pd

# models from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor



# metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score #mean_squared_error <- has been deprecated in version 1.4 and will be removed




##########################################################################################
####################################### REGRESSORS #######################################
##########################################################################################


######## LINEAR REGRESSION ########
def linear_regression(X_train, y_train, X_test, y_test):
    # make empty dataframe to store the metrics
    metrics = pd.DataFrame(columns=['Model', 'MAE', 'R2', 'EVS', 'RMSE']) # removed mse
    # make empty dictionary to store the best parameters
    best_params = {}

    # create an instance of the model
    model = LinearRegression()
    
    # fit the model
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)
    
    # metrics
    #mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Store metrics in the dataframe
    metrics.loc[len(metrics)] = ["lin reg", mae, r2, evs, rmse]
    metrics.to_csv('/work/DataScienceExam2024/Data/metrics_LinReg.csv', sep=';', index=False)

    print("The Linear regression is now fitted")
    return metrics

    
# Place to store all metrics
all_metrics = {}

######## GRID SEARCH AND MODEL FIT FOR EACH MODEL ######### 
def reg_model(model_class, X_train, y_train, X_test, y_test, param_grid_mod, kf):
    metrics = pd.DataFrame(columns=['Model', 'MAE', 'R2', 'EVS', 'RMSE']) # removed mse
    # make empty dictionary to store the best parameters
    best_params = {}

    print("Starting the grid search for", model_class.__name__)
    model = model_class()  # instantiate model
    grid = GridSearchCV(model, param_grid_mod, cv=kf, refit=True, verbose=1)

    # Fit model for grid search
    best_mod = grid.fit(X_train, y_train.squeeze())  # Use y_train.squeeze() to match expected dimensionality
    # save loss for each parameter combination
    cv_results = pd.DataFrame(best_mod.cv_results_)
    cv_results.to_csv(f'/work/DataScienceExam2024/Data_Science_Exam_S24/RegMod_Performance/CV_result_{model_class.__name__}.csv', sep=';', index=False)

    # Store the best parameters
    best_params[str(model_class)] = best_mod.best_params_
    # output best_params as csv
    # Convert the dictionary to a DataFrame and write csv 
    params_df = pd.DataFrame([{'Model': k, **v} for k, v in best_params.items()])
    params_df.to_csv(f'/work/DataScienceExam2024/Data_Science_Exam_S24/RegMod_Performance/BestParams_{model_class.__name__}.csv', sep=';', index=False)

    # Update model with best parameters
    mod = model_class(**best_mod.best_params_)
    mod.fit(X_train, y_train.squeeze())

    # Predict
    print("Predicting y with", model_class.__name__)
    y_pred = mod.predict(X_test)

    # Compute metrics
    print("Computing metrics for", model_class.__name__)
    #mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Store metrics in the dataframe
    metrics.loc[len(metrics)] = [model_class.__name__, mae, r2, evs, rmse]
    # save metrics as dataframe
    metrics.to_csv(f'/work/DataScienceExam2024/Data/metrics{model_class.__name__}.csv', sep=';', index=False)

    print(f"The fitting of {model_class.__name__} is done")
    return metrics




##########################################################################################
############################### DATA AND MODEL FIT #######################################
########################################################################################## 


param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

param_grid_ridge = {
    'alpha': [0.1, 1, 10, 100],
    'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}


param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# make param grid for MLP
param_grid_mlp = {
    'hidden_layer_sizes': [(50,),(100,),(50,50)],
    'activation': ['relu','logistic','tanh'],
    'solver': ['sgd','adam'],
    'alpha': [0.01, 0.001, 0.0001],
    'learning_rate': ['constant','adaptive'],
    'learning_rate_init': [0.001, 0.01]
}

# param grid knn
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30]
}

# param grid dt
param_grid_dt = {
    'criterion': ['poisson', 'friedman_mse', 'squared_error'],
    'splitter': ['best', 'random'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# combine param grids in dictionary
model_param_grids = {KNeighborsRegressor: param_grid_knn, 
               DecisionTreeRegressor: param_grid_dt,
               RandomForestRegressor: param_grid_rf,
               XGBRegressor: param_grid_xgb,
               MLPRegressor: param_grid_mlp,
               Ridge: param_grid_ridge
               }
               

# number of folds for the cross validation
kf = 5

# read all csv's from the training data - removing and old index column
X_train = pd.read_csv('/work/DataScienceExam2024/Data/X_train_scaled.csv', sep=',')
y_train = pd.read_csv('/work/DataScienceExam2024/Data/y_train.csv', sep=',')
X_test = pd.read_csv('/work/DataScienceExam2024/Data/X_val_scaled.csv', sep=',') # obs these are the validation set
y_test = pd.read_csv('/work/DataScienceExam2024/Data/y_val.csv', sep=',') # obs these are the validation set

# run the functions above
#linear_regression(X_train, y_train, X_test, y_test)
#print("Linear regression is done, starting on the loop of regressors")

# loop over models and param grids and run the function
for model_class, params in model_param_grids.items():
    reg_model(model_class, X_train, y_train, X_test, y_test, params, kf)


# save metrics as dataframe
#metrics.to_csv('/work/DataScienceExam2024/Data/metrics.csv', sep=';', index=False)

# save best parameters as dictionary
# Convert the dictionary to a DataFrame
#params_df = pd.DataFrame([{'Model': k, **v} for k, v in best_params.items()])
# Now use the DataFrame's `to_csv()` method to save it to a CSV file
#params_df.to_csv('/work/DataScienceExam2024/Data/best_params.csv', sep=';', index=False)

print("Job done!")