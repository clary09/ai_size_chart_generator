# src/accuracy_validation.py
from sklearn.metrics import mean_squared_error

class AccuracyValidator:
    def __init__(self, known_size_chart):
        self.known_size_chart = known_size_chart
    
    def validate(self, generated_size_chart):
        known_values = self.known_size_chart.values.flatten()
        generated_values = generated_size_chart.values.flatten()
        
        mse = mean_squared_error(known_values, generated_values)
        return mse
