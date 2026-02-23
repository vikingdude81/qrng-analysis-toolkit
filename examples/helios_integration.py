"""
HELIOS Integration Example
==========================

This example demonstrates how to integrate the anomaly scope
with a HELIOS neural network model.

Shows:
1. Creating torch tensors from model output
2. Passing batches to HeliosAnomalyScope
3. Real-time trajectory analysis during training/inference
"""

import numpy as np
import sys
sys.path.insert(0, '..')

# Note: torch is optional - this example works in simulation mode
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - running in simulation mode")


from helios_anomaly_scope import HeliosAnomalyScope


def create_mock_tensor(batch_size: int, features: int, device: str = 'cpu'):
    """Create a mock tensor simulating HELIOS output."""
    if TORCH_AVAILABLE:
        return torch.randn(batch_size, features, device=device)
    else:
        # Simulate tensor-like object
        class MockTensor:
            def __init__(self, data):
                self._data = data
            
            def cpu(self):
                return self
            
            def detach(self):
                return self
            
            def numpy(self):
                return self._data
            
            def mean(self, dim=None):
                return MockTensor(np.mean(self._data, axis=dim, keepdims=True))
            
            def std(self, dim=None):
                return MockTensor(np.std(self._data, axis=dim, keepdims=True))
            
            def __len__(self):
                return len(self._data)
        
        return MockTensor(np.random.randn(batch_size, features))


def run_helios_integration(num_batches: int = 100, batch_size: int = 32,
                            features: int = 64, inject_drift_at: int = None):
    """
    Demonstrate HELIOS integration.
    
    Args:
        num_batches: Number of batches to process
        batch_size: Size of each batch
        features: Number of features per sample
        inject_drift_at: Batch at which to inject drift
    """
    print("=" * 60)
    print("HELIOS Integration Example")
    print("=" * 60)
    
    # Create the anomaly scope
    scope = HeliosAnomalyScope(
        history_len=50,      # Keep 50 batch summaries
        msd_window=20        # Window for MSD analysis
    )
    
    print(f"\n📊 Configuration:")
    print(f"   Batches: {num_batches}")
    print(f"   Batch size: {batch_size}")
    print(f"   Features: {features}")
    print(f"   Torch available: {TORCH_AVAILABLE}")
    if inject_drift_at:
        print(f"   ⚠️  Drift injection at batch {inject_drift_at}")
    
    print(f"\n🔄 Processing batches...")
    
    for batch_idx in range(num_batches):
        # Create batch (with optional drift)
        if inject_drift_at and batch_idx >= inject_drift_at:
            # Add drift - shift mean
            if TORCH_AVAILABLE:
                active_batch = torch.randn(batch_size, features) + 0.5
            else:
                data = np.random.randn(batch_size, features) + 0.5
                active_batch = create_mock_tensor(batch_size, features)
                active_batch._data = data
        else:
            active_batch = create_mock_tensor(batch_size, features)
        
        # Process through scope
        metrics = scope.update(active_batch)
        
        # Skip while building history
        if metrics.get('waiting_for_history'):
            continue
        
        # Print progress every 20 batches
        if batch_idx > 0 and batch_idx % 20 == 0:
            print(f"   Batch {batch_idx}: H={metrics.get('hurst', 0):.3f}, "
                  f"λ={metrics.get('lyapunov', 0):.3f}, "
                  f"influence={metrics.get('influence_detected', False)}")
    
    # Final results
    print(f"\n📈 Results:")
    
    events = scope.events
    print(f"   Total events: {len(events)}")
    
    # Check verified events
    verified = scope.get_verified_events()
    print(f"   Verified events: {len(verified)}")
    
    # Signal verification
    verification = scope.verify_current_signal()
    if verification:
        print(f"\n🔍 Signal Verification:")
        print(f"   Classification: {verification.signal_class.name}")
        print(f"   Is verified: {verification.is_verified}")
        print(f"   Confidence: {verification.confidence:.2f}")
    
    return scope


class HeliosModelWrapper:
    """
    Example wrapper showing how to integrate with a real HELIOS model.
    
    This wraps a model and automatically analyzes its activations.
    """
    
    def __init__(self, model, layer_name: str = 'output'):
        """
        Args:
            model: A PyTorch model
            layer_name: Name of layer to monitor
        """
        self.model = model
        self.layer_name = layer_name
        self.scope = HeliosAnomalyScope(history_len=100)
        self._activations = None
        
        # Register hook if torch is available
        if TORCH_AVAILABLE and hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                if name == layer_name:
                    module.register_forward_hook(self._hook)
                    break
    
    def _hook(self, module, input, output):
        """Hook to capture activations."""
        self._activations = output
    
    def __call__(self, x):
        """Forward pass with analysis."""
        output = self.model(x)
        
        if self._activations is not None:
            metrics = self.scope.update(self._activations)
            
            if metrics.get('influence_detected'):
                print(f"⚠️  Anomaly detected in {self.layer_name}!")
        
        return output
    
    def get_analysis(self):
        """Get current analysis state."""
        return {
            'events': self.scope.events,
            'verification': self.scope.verify_current_signal(),
            'classification': self.scope.get_signal_classification()
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HELIOS Integration Example')
    parser.add_argument('--batches', type=int, default=100, help='Number of batches')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--inject-drift', type=int, default=None,
                        help='Batch at which to inject drift')
    
    args = parser.parse_args()
    
    run_helios_integration(
        num_batches=args.batches,
        batch_size=args.batch_size,
        inject_drift_at=args.inject_drift
    )
