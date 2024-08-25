# main.py
from src.data_preprocessing import DataPreprocessor
from src.accuracy_validation import AccuracyValidator
from src.category_clustering import CategoryClustering
from src.size_chart import SizeChartGenerator
from src.return_simulation import ReturnSimulation
from src.transfer_learning import TransferLearning
from src.batch_processing import BatchProcessor
from src.database import Database
import pandas as pd

# Initialize components
preprocessor = DataPreprocessor('data/user_data.csv')
category_clustering = CategoryClustering(num_clusters=5)
size_chart_generator = SizeChartGenerator(size_labels={0: 'S', 1: 'M', 2: 'L', 3: 'XL', 4: 'XXL'})
batch_processor = BatchProcessor(batch_size=100)

# Load known size chart for accuracy validation
known_size_chart = pd.read_csv('data/known_size_chart.csv')
accuracy_validator = AccuracyValidator(known_size_chart)

# Load and preprocess data
db = Database('postgresql://user:password@localhost/size_chart_db')
data = db.get_all_user_data()
data_normalized = preprocessor.preprocess(data)

# Handle different apparel categories
for category in ['tops', 'bottoms', 'dresses']:
    category_data = data_normalized[data_normalized['category'] == category]
    clusters = category_clustering.fit_predict(category_data, category)
    size_chart = size_chart_generator.generate_size_chart(category_data, clusters)
    
    # Validate accuracy
    mse = accuracy_validator.validate(size_chart)
    print(f'Category: {category}, MSE: {mse}')
    
    # Batch processing for efficiency
    batch_results = batch_processor.process_in_batches(category_data, lambda batch: category_clustering.fit_predict(batch, category))
    
# Simulate return rates
holdout_data = pd.read_csv('data/holdout_data.csv')
return_simulator = ReturnSimulation(holdout_data, size_chart_generator, category_clustering)
return_rate = return_simulator.simulate_returns()
print(f'Return Rate: {return_rate}')

# Adapt to new brands using transfer learning
new_brand_data = pd.read_csv('data/new_brand_data.csv')
transfer_learning = TransferLearning(category_clustering)
adapted_model = transfer_learning.adapt_to_new_brand(new_brand_data)
