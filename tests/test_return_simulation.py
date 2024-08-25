import pytest
from return_simulation import ReturnSimulator

@pytest.fixture
def mock_purchase_data():
    return [
        {'size': 'M', 'returned': False},
        {'size': 'L', 'returned': True},
        {'size': 'M', 'returned': False},
        {'size': 'L', 'returned': False}
    ]

def test_return_simulation(mock_purchase_data):
    simulator = ReturnSimulator(mock_purchase_data)
    return_rate = simulator.simulate()
    assert return_rate < 0.5, f"Return rate too high: {return_rate}"

def test_simulation_with_new_data():
    new_data = [
        {'size': 'M', 'returned': True},
        {'size': 'S', 'returned': False},
        {'size': 'M', 'returned': True}
    ]
    simulator = ReturnSimulator(new_data)
    return_rate = simulator.simulate()
    assert return_rate > 0.5, "Simulation did not handle new data as expected"
