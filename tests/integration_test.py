import os
import pytest
from helios.experiments import ExperimentConfig, ExperimentManager

def test_integration():
    config_path = 'test_config.yaml'
    with open(config_path, 'w') as f:
        f.write('seed: 12345\n')
    
    config = ExperimentConfig(config_path)
    manager = ExperimentManager(config)
    manager.run_experiment()
    
    assert os.path.exists('results.hdf5')
    with File('results.hdf5', 'r') as hf:
        result = hf['result'][()]
        assert np.allclose(result, np.random.rand(10))
