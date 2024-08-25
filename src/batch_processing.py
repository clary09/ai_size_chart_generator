# src/batch_processing.py
class BatchProcessor:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
    
    def process_in_batches(self, data, process_function):
        num_batches = len(data) // self.batch_size + (1 if len(data) % self.batch_size != 0 else 0)
        results = []
        
        for i in range(num_batches):
            batch = data[i*self.batch_size : (i+1)*self.batch_size]
            batch_result = process_function(batch)
            results.append(batch_result)
        
        return results
