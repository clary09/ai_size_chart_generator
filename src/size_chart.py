# src/size_chart.py
import pandas as pd

class SizeChartGenerator:
    def __init__(self, size_labels):
        self.size_labels = size_labels
    
    def generate_size_chart(self, data, clusters):
        size_chart = pd.DataFrame(columns=['size', 'height', 'weight', 'chest', 'waist', 'hip'])
        
        for cluster in range(len(self.size_labels)):
            cluster_data = data[clusters == cluster]
            size_chart.loc[cluster] = [
                self.size_labels[cluster],
                cluster_data['height'].mean(),
                cluster_data['weight'].mean(),
                cluster_data['chest'].mean(),
                cluster_data['waist'].mean(),
                cluster_data['hip'].mean()
            ]
        
        return size_chart
