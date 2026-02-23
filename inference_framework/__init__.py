"""
Inference Architectures: A Framework for Multi-Modal Reasoning

Based on the Intuition Machine framework extending Peirce's classical triad
(abduction, deduction, induction) into 12 modes organized around three
fundamental architectures of validity-production.

The Three Architectures:
- Strange Attractor: Convergent, monadic inference (insight through convergence)
- Code Duality: Coupled, dyadic inference (certainty through perfect coupling)
- Tensegrity: Balanced, tetradic inference (reasonableness through balanced forces)

Main Components:
- architectures: Core architecture implementations
- classifier: Mode classification and routing
- experiment: QRNG vs PRNG experimental framework
- qrng_bridge: Integration with helios QRNG data streams
"""

from .architectures import (
    Architecture,
    InferenceMode,
    MODE_TO_ARCHITECTURE,
    InferenceState,
    InferenceResult,
    BaseArchitecture,
    StrangeAttractorArchitecture,
    CodeDualityArchitecture,
    TensegrityArchitecture,
)

from .classifier import (
    ClassificationResult,
    ModeClassifier,
    ReasoningRouter,
)

from .experiment import (
    RandomnessSource,
    TrialResult,
    TaskDefinition,
    ExperimentConfig,
    RandomnessProvider,
    InferenceExperiment,
    STANDARD_TASKS,
)

from .qrng_bridge import (
    QRNGSourceType,
    StreamStats,
    QRNGStreamProvider,
    LiveOutshiftProvider,
    ANUQRNGProvider,
    CipherstoneQRNGMode,
    CipherstoneQRNGProvider,
    UnifiedRandomnessProvider,
    create_inference_experiment_with_qrng,
)

__version__ = "0.1.0"
__author__ = "Alex"

__all__ = [
    # Architectures
    "Architecture",
    "InferenceMode",
    "MODE_TO_ARCHITECTURE",
    "InferenceState",
    "InferenceResult",
    "BaseArchitecture",
    "StrangeAttractorArchitecture",
    "CodeDualityArchitecture",
    "TensegrityArchitecture",
    
    # Classifier
    "ClassificationResult",
    "ModeClassifier",
    "ReasoningRouter",
    
    # Experiment
    "RandomnessSource",
    "TrialResult",
    "TaskDefinition",
    "ExperimentConfig",
    "RandomnessProvider",
    "InferenceExperiment",
    "STANDARD_TASKS",
    
    # QRNG Bridge (helios integration)
    "QRNGSourceType",
    "StreamStats",
    "QRNGStreamProvider",
    "LiveOutshiftProvider",
    "ANUQRNGProvider",
    "CipherstoneQRNGMode",
    "CipherstoneQRNGProvider",
    "UnifiedRandomnessProvider",
    "create_inference_experiment_with_qrng",
]
