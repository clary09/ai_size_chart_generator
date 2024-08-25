import pytest
from transfer_learning import TransferLearning

@pytest.fixture
def base_model():
    return mock_pretrained_model()  # Assuming a function that returns a pre-trained model

@pytest.fixture
def new_brand_data():
    return pd.DataFrame({
        'height': [165, 175, 185],
        'weight': [55, 75, 95],
        'chest': [85, 95, 105],
        'waist': [65, 75, 85],
        'hip': [95, 105, 115]
    })

def test_transfer_learning(base_model, new_brand_data):
    transfer_learner = TransferLearning(base_model)
    adapted_model = transfer_learner.adapt(new_brand_data)
    predictions = adapted_model.predict(new_brand_data)
    assert predictions is not None, "Transfer learning did not produce predictions"

def test_transfer_learning_efficiency(base_model, new_brand_data):
    transfer_learner = TransferLearning(base_model)
    adapted_model = transfer_learner.adapt(new_brand_data)
    initial_predictions = base_model.predict(new_brand_data)
    adapted_predictions = adapted_model.predict(new_brand_data)
    assert initial_predictions != adapted_predictions, "Transfer learning did not significantly adapt the model"
