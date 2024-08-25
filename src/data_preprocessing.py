# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def preprocess(self, data):
        # Normalize numerical data
        scaler = StandardScaler()
        numerical_columns = ['height', 'weight', 'chest', 'waist', 'hip']
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        return data
