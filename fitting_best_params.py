# IMPORT PACKAGES   
from sklearn.dummy import DummyRegressor
import pandas as pd
import numpy as np

# models from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler


# metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score #mean_squared_error <- has been deprecated in version 1.4 and will be removed

# read all csv's from the training data - removing and old index column
X_train = pd.read_csv('/work/exam_repo/Data/X_train_scaled_80.csv', sep=',')
y_train = pd.read_csv('/work/exam_repo/Data/y_train_20.csv', sep=',')
X_test = pd.read_csv('/work/exam_repo/Data/X_test_scaled_80.csv', sep=',') 
y_test = pd.read_csv('/work/exam_repo/Data/y_test_20.csv', sep=',') 

### df to store in: 

metrics = pd.DataFrame(columns=['Model', 'MAE', 'R2', 'EVS', 'RMSE']) # removed mse
predictions = pd.DataFrame(columns=['Model', 'Predicted Values', 'True Values'])



##########################################################################################
############################## GETTING PREDICTIONS FOR BASELINES ####################
##########################################################################################
# predicting the mean of the target variable
def dummy_mean(X_train, y_train, X_test, y_test): 
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)

    # Ensure y_test is a 1D numpy array
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze().to_numpy()
    elif isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    
    # Store metrics in the dataframe
    global metrics
    metrics.loc[len(metrics)] = ["dummy_mean", mae, r2, evs, rmse]

    # storing predictions for each model
    data_to_append = pd.DataFrame({
        'Model': np.repeat("dummy_mean", len(y_pred)),  # Repeat model name for each entry
        'Predicted Values': y_pred,
        'True Values': y_test
    })

    # Use concat to add the new rows to the DataFrame
    global predictions
    predictions = pd.concat([predictions, data_to_append], ignore_index=True)



# run the function above
dummy_mean(X_train, y_train, X_test, y_test)
print("The mean dummy model is finished!")



#### manually doing it instead: 
def dummy_yesterday(X_train, y_train, X_test, y_test):
    print("The Dummy_Yesterday model is now fitting")
    
    #  Extract the 'Antal_timer_yesterday' as a numpy array and reshape it for scaling
    y_pred = X_test['Antal_timer_yesterday'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.min_ = np.array([y_train.min()])  # min of the feature
    scaler.scale_ = np.array([1/(y_train.max() - y_train.min())])  # scale of the feature 

    # Apply inverse transformation to get back to original values
    y_pred_original = scaler.inverse_transform(y_pred)

    # Ensure y_test is a 1D numpy array
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze().to_numpy()
    elif isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    # metrics
    mae = mean_absolute_error(y_test, y_pred_original)
    rmse = root_mean_squared_error(y_test, y_pred_original)
    r2 = r2_score(y_test, y_pred_original)
    evs = explained_variance_score(y_test, y_pred_original)

    # Store metrics in the dataframe
    global metrics
    metrics.loc[len(metrics)] = ["dummy_yesterday", mae, r2, evs, rmse]
    

    data_to_append = pd.DataFrame({
        'Model': np.repeat("dummy_yesterday", len(y_pred_original)),  # Repeat model name for each entry
        'Predicted Values': y_pred_original.flatten(),
        'True Values': y_test
    })

    # Use concat to add the new rows to the DataFrame
    global predictions
    predictions = pd.concat([predictions, data_to_append], ignore_index=True)

    return metrics, predictions

dummy_yesterday(X_train, y_train, X_test, y_test)
print("The dummy model predicting yesterdays outcome is finished!")

######## LINEAR REGRESSION ########
def linear_regression(X_train, y_train, X_test, y_test):

    # create an instance of the model
    model = LinearRegression()
    
    # fit the model
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)

    # Ensure y_test is a 1D numpy array
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze().to_numpy()
    elif isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()
    
    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Store metrics in the dataframe
    global metrics
    metrics.loc[len(metrics)] = ["lin_reg", mae, r2, evs, rmse]

    # making sure the y_pred and test is fittable in the dataframe as it struggles to run
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()  # Flatten y_pred to 1D if it's 2D
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze().to_numpy()  # Squeeze DataFrame to 1D numpy array
    elif isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()  # Convert Series to numpy array
    elif y_test.ndim > 1:
        y_test = y_test.ravel()  # Flatten y_test to 1D if it's 2D

    # storing predictions for each model 
    data_to_append = pd.DataFrame({
        'Model': np.repeat("lin_reg", len(y_pred)),  # Repeat model name for each entry
        'Predicted Values': y_pred,
        'True Values': y_test
    })

    # Use concat to add the new rows to the DataFrame
    global predictions
    predictions = pd.concat([predictions, data_to_append], ignore_index=True)

    
    print("The Linear regression is now fitted")

    return metrics, predictions


# run the function above
linear_regression(X_train, y_train, X_test, y_test)

##########################################################################################
############################## FITTIING MODELS TO BEST PARAMS FOR REG ####################
##########################################################################################

def model_fit(model_class, X_train, x_test, y_train, y_test):
    
    model = model_class()  # instantiate model

    # read the best parameters found in the grid search: 
    best_param = pd.read_csv(f'/work/exam_repo/PredictingAbsence_DataScience2024/RegMod_Performance/BestParameters/BestParams_{model_class.__name__}.csv', sep=';')
    best_param.drop(columns=['Model'], inplace=True)
    best_param_dict = best_param.to_dict(orient='records')[0]

    # Special handling for hidden_layer_sizes which is expected to be a tuple
    if 'hidden_layer_sizes' in best_param_dict:
        # Parse the string from CSV back to a tuple
        best_param_dict['hidden_layer_sizes'] = literal_eval(best_param_dict['hidden_layer_sizes'])


    # fitting w best params 
    mod = model_class(**best_param_dict)
    mod.fit(X_train, y_train.squeeze())

    y_pred = mod.predict(X_test)

    # Ensure y_test is a 1D numpy array
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze().to_numpy()
    elif isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    # storing metrics just in case
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    global metrics
    metrics.loc[len(metrics)] = [model_class.__name__, mae, r2, evs, rmse]

    # storing predictions for each model
    data_to_append = pd.DataFrame({
        'Model': np.repeat(model_class.__name__, len(y_pred)),  # Repeat model name for each entry
        'Predicted Values': y_pred,
        'True Values': y_test
    })

    # Use concat to add the new rows to the DataFrame
    global predictions
    predictions = pd.concat([predictions, data_to_append], ignore_index=True)

    return metrics, predictions
    


models = [Ridge, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, MLPRegressor]

# loop over models and param grids and run the function
for model_class in models:
    model_fit(model_class, X_train,X_test, y_train, y_test)
    print(f"The {model_class} is finished!")



metrics.to_csv(f'/work/exam_repo/PredictingAbsence_DataScience2024/RegMod_Performance/metrics_test_latest.csv', sep=';', index=False)
predictions.to_csv(f'/work/exam_repo/PredictingAbsence_DataScience2024/RegMod_Performance/predictions_test_latest.csv', sep=';', index=False)

### getting the first 100 true values for plotting purposes: 
first_predictions = pd.DataFrame(predictions['True Values'][0:100])
first_predictions.to_csv(f'/work/exam_repo/PredictingAbsence_DataScience2024/plots/true100val.csv', index=False)