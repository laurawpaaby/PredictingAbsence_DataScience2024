
from sklearn.dummy import DummyRegressor
import pandas as pd
import numpy as np

# metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score #mean_squared_error <- has been deprecated in version 1.4 and will be removed


########### MEAN MODEL ###########
# making a model that is predicting the mean of the target variable
def dummy_mean(X_train, y_train, X_test, y_test): 
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Store metrics in the dataframe
    metrics = pd.DataFrame(columns=['Model', 'MAE', 'R2', 'EVS', 'RMSE'])
    metrics.loc[len(metrics)] = ["dummy_mean", mae, r2, evs, rmse]
    metrics.to_csv('/work/DataScienceExam2024/Data/metrics_Dummy_Mean.csv', sep=';', index=False)

    print("The Dummy_Mean model is now fitted")
    return metrics



########### THE DAY BEFORE MODEL ############################
def dummy_yesterday(X_test, y_test):
    print("The Dummy_Yesterday model is now fitting")
    # Create an instance of the custom predictor
    y_pred = X_test['Antal_timer_yesterday']

    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Store metrics in the dataframe
    metrics = pd.DataFrame(columns=['Model', 'MAE', 'R2', 'EVS', 'RMSE'])
    metrics.loc[len(metrics)] = ["dummy_yesterday", mae, r2, evs, rmse]
    metrics.to_csv('/work/DataScienceExam2024/Data/metrics_Dummy_Yesterday.csv', sep=';', index=False)

    return metrics




#### run the models
# read all csv's from the training data - removing and old index column
X_train = pd.read_csv('/work/DataScienceExam2024/Data/X_train_scaled.csv', sep=',')
y_train = pd.read_csv('/work/DataScienceExam2024/Data/y_train.csv', sep=',')
X_test = pd.read_csv('/work/DataScienceExam2024/Data/X_val_scaled.csv', sep=',') # obs these are the validation set
y_test = pd.read_csv('/work/DataScienceExam2024/Data/y_val.csv', sep=',') # obs these are the validation set




# run the functions above
dummy_mean(X_train, y_train, X_test, y_test)
print("The mean dummy model is finished!")

# run the functions above
dummy_yesterday(X_test, y_test)
print("The dummy model predicting yesterdays outcome is finished!")