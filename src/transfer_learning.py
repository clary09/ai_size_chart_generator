# src/transfer_learning.py
from sklearn.base import clone

class TransferLearning:
    def __init__(self, base_model):
        self.base_model = base_model
    
    def adapt_to_new_brand(self, new_data):
        new_model = clone(self.base_model)
        new_model.fit(new_data)
        return new_model
