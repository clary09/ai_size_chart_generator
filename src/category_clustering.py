# src/category_clustering.py
from sklearn.mixture import GaussianMixture

class CategoryClustering:
    def __init__(self, num_clusters=5):
        self.models = {
            "tops": GaussianMixture(n_components=num_clusters),
            "bottoms": GaussianMixture(n_components=num_clusters),
            "dresses": GaussianMixture(n_components=num_clusters)
        }
    
    def fit_predict(self, data, category):
        model = self.models[category]
        clusters = model.fit_predict(data)
        return clusters
    
    def update_model(self, data, category):
        model = self.models[category]
        clusters = model.fit_predict(data)
        return clusters
