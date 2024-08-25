# src/return_simulation.py
class ReturnSimulation:
    def __init__(self, holdout_data, size_chart_generator, clustering_model):
        self.holdout_data = holdout_data
        self.size_chart_generator = size_chart_generator
        self.clustering_model = clustering_model
    
    def simulate_returns(self):
        successful_fits = 0
        total_purchases = len(self.holdout_data)
        
        for _, row in self.holdout_data.iterrows():
            category = row['category']
            cluster = self.clustering_model.fit_predict([row['measurements']], category)
            size_chart = self.size_chart_generator.generate_size_chart(row['measurements'], cluster)
            predicted_size = size_chart.loc[cluster, 'size']
            
            if predicted_size == row['actual_size']:
                successful_fits += 1
        
        return_rate = 1 - (successful_fits / total_purchases)
        return return_rate
