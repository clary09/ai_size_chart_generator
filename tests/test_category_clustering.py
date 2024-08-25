import pytest
import pandas as pd
from category_clustering import CategoryClustering

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'height': [160, 170, 180, 190],
        'weight': [50, 70, 90, 110],
        'chest': [80, 90, 100, 110],
        'waist': [60, 70, 80, 90],
        'hip': [90, 100, 110, 120]
    })

def test_category_clustering(sample_data):
    clusterer = CategoryClustering(n_clusters=2)
    clusters = clusterer.fit_predict(sample_data)
    assert len(set(clusters)) == 2, "Clustering did not produce the expected number of clusters"

def test_clustering_edge_cases():
    edge_case_data = pd.DataFrame({
        'height': [160, 160, 160],
        'weight': [50, 50, 50],
        'chest': [80, 80, 80],
        'waist': [60, 60, 60],
        'hip': [90, 90, 90]
    })
    clusterer = CategoryClustering(n_clusters=1)
    clusters = clusterer.fit_predict(edge_case_data)
    assert len(set(clusters)) == 1, "Clustering failed on edge case"
