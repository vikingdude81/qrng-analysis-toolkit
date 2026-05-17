# Integration Tests for helios_anomaly_scope.py

import pytest
from tests.synthetic_data_generator import generate_anomaly_data
from helios.anomaly_scope import AnomalyScopeDetector

def test_anomaly_detection():
    # Generate synthetic data with anomalies
    data = generate_anomaly_data()
    
    # Initialize detector with appropriate parameters
    detector = AnomalyScopeDetector(threshold=0.1)
    
    # Detect anomalies
    detected = detector.detect(data)
    
    # Verify recall and FPR
    assert detected['recall'] > 0.95, "Recall below 95%"
    assert detected['fpr'] < 0.01, "FPR above 1%"

@pytest.mark.parametrize("anomaly_type", ["entropy_drop", "spike", "non_stationarity"])
def test_anomaly_detection_with_type(anomaly_type):
    data = generate_anomaly_data(anomaly_type=anomaly_type)
    detector = AnomalyScopeDetector(threshold=0.1)
    detected = detector.detect(data)
    assert detected['recall'] > 0.95, f"Recall failed for {anomaly_type}"
    assert detected['fpr'] < 0.01, f"FPR failed for {anomaly_type}"

# Run full test suite
pytest helios/ --cov=helios --cov-report=term-missing -v