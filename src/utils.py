import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def normalize_data(data):
    """
    Normalize the input data using StandardScaler.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data to be normalized.
    
    Returns:
        pd.DataFrame: Normalized data as a DataFrame.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def calculate_mse(actual, predicted):
    """
    Calculate the Mean Squared Error (MSE) between actual and predicted values.
    
    Args:
        actual (np.array or pd.Series): Actual values.
        predicted (np.array or pd.Series): Predicted values.
    
    Returns:
        float: Mean Squared Error between actual and predicted values.
    """
    return mean_squared_error(actual, predicted)

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data to be split.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given set of data.
    
    Args:
        data (np.array or pd.Series): Data for which to calculate the confidence interval.
        confidence (float): Confidence level for the interval.
    
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    data_mean = np.mean(data)
    data_std = np.std(data)
    n = len(data)
    margin_of_error = data_std / np.sqrt(n) * 1.96  # Z-value for 95% confidence
    lower_bound = data_mean - margin_of_error
    upper_bound = data_mean + margin_of_error
    return data_mean, lower_bound, upper_bound

def save_model(model, filepath):
    """
    Save a machine learning model to a file.
    
    Args:
        model (sklearn model): The model to save.
        filepath (str): The file path where the model should be saved.
    """
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a machine learning model from a file.
    
    Args:
        filepath (str): The file path from which to load the model.
    
    Returns:
        sklearn model: The loaded model.
    """
    import joblib
    return joblib.load(filepath)

def get_feature_importances(model, feature_names):
    """
    Get feature importances from a trained model.
    
    Args:
        model (sklearn model): The trained model with feature importances.
        feature_names (list of str): List of feature names.
    
    Returns:
        pd.DataFrame: DataFrame with feature names and their corresponding importances.
    """
    importances = model.feature_importances_
    return pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (sklearn model): The trained model to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
    
    Returns:
        dict: A dictionary containing evaluation metrics (e.g., accuracy, MSE).
    """
    predictions = model.predict(X_test)
    mse = calculate_mse(y_test, predictions)
    return {
        'Mean Squared Error': mse,
        # Add more metrics if needed
    }
