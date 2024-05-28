import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor



def evaluate_model(model_class, X_train, y_train, X_test, y_test):
    # Validate that the model_class is either RandomForestRegressor or XGBRegressor
    if model_class not in [RandomForestRegressor, XGBRegressor]:
        raise ValueError("Model must be either RandomForestRegressor or XGBRegressor")

    # File path to the CSV containing best parameters
    file_path = f'/work/DataScienceExam2024/Data_Science_Exam_S24/RegMod_Performance/BestParams_{model_class.__name__}.csv'
    
    # Read the best parameters found in the grid search
    best_param = pd.read_csv(file_path, sep=';')
    best_param.drop(columns=['Model'], inplace=True)
    best_param_dict = best_param.to_dict(orient='records')[0]

    # Initialize and fit the model with the best parameters
    model = model_class(**best_param_dict)
    model.fit(X_train, y_train.squeeze())

    # Perform permutation importance analysis
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

    # Create a DataFrame for importances
    df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })

    # Save the DataFrame to a CSV file
    df.to_csv(f'/work/DataScienceExam2024/Data_Science_Exam_S24/RegMod_Performance/feature_importances_{model_class.__name__}.csv', index=False)

    return df


### loading data: 
X_train = pd.read_csv('/work/DataScienceExam2024/Data/X_train_scaled.csv', sep=',')
y_train = pd.read_csv('/work/DataScienceExam2024/Data/y_train.csv', sep=',')
X_test = pd.read_csv('/work/DataScienceExam2024/Data/X_test_scaled.csv', sep=',') 
y_test = pd.read_csv('/work/DataScienceExam2024/Data/y_test.csv', sep=',') 

# running the func 
evaluate_model(RandomForestRegressor, X_train, y_train, X_test, y_test)
evaluate_model(XGBRegressor, X_train, y_train, X_test, y_test)
