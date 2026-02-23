# Multi-Modal Inference Architectures

A framework for implementing the 12-mode inference system based on the Intuition Machine's extension of Peirce's classical triad. Includes an experimental framework for testing entropy sensitivity across inference architectures.

## Overview

This project implements three fundamental "architectures of validity-production":

### 1. Strange Attractor (Convergence)
Monadic, self-referential processes that settle into stable "basins of attraction."

**Modes:**
- **Mode 1: Abduction** - "What would explain this?" - Iterative hypothesis generation
- **Mode 9: Qualitative** - "What do qualities intimate?" - Pre-conceptual felt-sense reasoning  
- **Mode 12: Equilibrium** - "Where does the process converge?" - Fixed-point analysis

**Key Characteristics:**
- Driven by "immediate attraction for the idea itself"
- Process is iterative and clarifying
- Conclusion is a stable "fixed point" of understanding

### 2. Code Duality (Coupling)
Dyadic, deterministic mappings between two domains (e.g., proof and program, type and token).

**Modes:**
- **Mode 3: Deduction** - "What must follow?" - Necessary inference
- **Mode 4: Syntactic** - "How is it organized?" - Structural analysis
- **Mode 5: Diagrammatic** - "What does manipulation reveal?" - Visual/formal reasoning
- **Mode 8: Metonymic** - "What does convention connect?" - Cultural association
- **Mode 11: Type-Token** - "What does instantiation transfer?" - Property inheritance

**Key Characteristics:**
- Based on mutual, deterministic mapping
- Validity is absolute within the system
- Represents the "necessary inference" aspect of reason

### 3. Tensegrity (Balance)
Tetradic structures held in productive tension, requiring at least four elements.

**Modes:**
- **Mode 2: Induction** - "What pattern holds?" - Generalization with counter-examples
- **Mode 6: Analogical** - "What corresponds across domains?" - Structural mapping
- **Mode 7: Indexical** - "What does this trace show?" - Effect→cause tracking
- **Mode 10: Contrastive** - "What does difference reveal?" - Reasoning through opposition

**Key Characteristics:**
- Maintains stability through balanced opposition
- The structure itself is the source of validity
- Represents the "reasonableness" aspect of inference

## Installation

```bash
cd /path/to/project
pip install -r requirements.txt  # (if requirements.txt exists)
```

## Quick Start

```python
from inference_architectures.core import ReasoningRouter, ModeClassifier

# Initialize with your LLM
def my_llm_call(prompt: str) -> str:
    # Your LLM API call here
    pass

router = ReasoningRouter(llm_call=my_llm_call)

# Automatic mode detection and routing
result = router.reason(
    task="What would explain the sudden drop in user engagement?",
    inputs={"context": "SaaS product analytics"}
)

print(f"Mode: {result['classification']['mode']}")
print(f"Conclusion: {result['result']['conclusion']}")
```

## Running the Demo

```bash
python demo.py
```

## Experimental Framework

The project includes a framework for testing whether different inference architectures show differential sensitivity to the source of randomness (QRNG vs PRNG).

### Hypothesis

Strange Attractor and Tensegrity architectures will show measurable differences in convergence dynamics when using quantum random number generators (QRNG) compared to pseudo-random number generators (PRNG), while Code Duality (being more deterministic) will show no significant difference.

### Running Experiments

```python
from inference_architectures.core import (
    InferenceExperiment,
    ExperimentConfig,
    STANDARD_TASKS
)

config = ExperimentConfig(
    experiment_id="entropy_sensitivity_001",
    trials_per_condition=10,
    max_iterations=10
)

experiment = InferenceExperiment(
    llm_call=my_llm_call,
    config=config,
    qrng_source=my_qrng_function  # Optional: actual QRNG hardware
)

results = experiment.run_experiment(tasks=STANDARD_TASKS)
print(results['analysis']['hypothesis_test'])
```

### Connecting QRNG Hardware

To use actual quantum random numbers:

```python
# Example: ANU QRNG API
import requests

def anu_qrng() -> float:
    response = requests.get(
        "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint16"
    )
    data = response.json()
    return data['data'][0] / 65535.0

experiment = InferenceExperiment(
    llm_call=my_llm_call,
    config=config,
    qrng_source=anu_qrng
)
```

---

## Paper Outline

**Title:** *Randomness Source Effects on Multi-Modal Inference: Characterizing Entropy Sensitivity in Reasoning Architectures*

### Abstract
We investigate whether different inference architectures exhibit differential sensitivity to the source of randomness in their computational processes. Building on the Intuition Machine framework that extends Peirce's classical inferential triad into 12 distinct modes organized by three architectures—Strange Attractor (convergence), Code Duality (coupling), and Tensegrity (balance)—we hypothesize that architectures involving iterative convergence and dynamic balance will show greater sensitivity to quantum vs. pseudo-random number sources than deterministic coupling architectures.

### 1. Introduction
- Peirce's triad and its limitations
- The 12-mode extension: from static logic to dynamic process architectures
- Why randomness source might matter for iterative inference

### 2. Theoretical Framework
#### 2.1 The Three Architectures
- Strange Attractor: Monadic, self-referential convergence
- Code Duality: Dyadic, deterministic coupling
- Tensegrity: Tetradic, balanced opposition

#### 2.2 Role of Randomness in Each Architecture
- Strange Attractor: Exploration during hypothesis search
- Code Duality: Rule selection (minimal role)
- Tensegrity: Force rebalancing during tension resolution

#### 2.3 Hypothesis
Convergent (Strange Attractor) and balanced (Tensegrity) architectures will show larger effect sizes for QRNG vs PRNG differences than deterministic (Code Duality) architectures.

### 3. Methods
#### 3.1 Implementation
- LLM-based instantiation of each inference mode
- Randomness injection points
- Convergence detection

#### 3.2 Experimental Design
- 12 standard tasks (4 per architecture)
- 3 randomness conditions: QRNG, PRNG-secure, PRNG-fast
- N trials per condition
- Dependent variables: iterations to convergence, solution diversity, confidence trajectory

#### 3.3 QRNG Integration
- Hardware specifications
- Comparison to cryptographic PRNG
- Statistical validation of randomness sources

### 4. Results
#### 4.1 By Architecture
- Strange Attractor: [results]
- Code Duality: [results]
- Tensegrity: [results]

#### 4.2 Interaction Effects
- Architecture × Randomness Source ANOVA
- Effect sizes (Cohen's d) for QRNG vs PRNG by architecture

#### 4.3 Solution Diversity Analysis
- Do QRNG runs produce more diverse hypotheses?
- Path entropy through convergence

### 5. Discussion
#### 5.1 Implications for AI System Design
- When does true randomness matter?
- Computational cost vs. epistemic benefit

#### 5.2 Theoretical Contributions
- Formalizing inference as dynamic process
- Connecting entropy to epistemic architecture

#### 5.3 Limitations and Future Work
- LLM as inference substrate
- Task representativeness
- Causal mechanism unclear

### 6. Conclusion
[Summary of findings and significance]

### References
- Peirce's original work on inference
- Intuition Machine framework
- QRNG in computational systems
- LLM reasoning capabilities

---

## Project Structure

```
inference_architectures/
├── core/
│   ├── __init__.py          # Package exports
│   ├── architectures.py     # Base classes and 3 architecture implementations
│   ├── classifier.py        # Mode classification and routing
│   └── experiment.py        # QRNG experiment framework
├── demo.py                  # Demonstration script
└── README.md               # This file
```

## Contributing

This is an experimental framework. Key areas for contribution:

1. **Real QRNG integration**: Connect to actual quantum hardware
2. **Improved classification**: Better pattern matching and LLM-based classification
3. **Additional modes**: Implement remaining modes in detail
4. **Semantic convergence**: Better detection of hypothesis/conclusion similarity
5. **Visualization**: Path through hypothesis space, tension landscapes

## License

MIT License

## Citation

If you use this framework, please cite:

```bibtex
@software{inference_architectures,
  title={Multi-Modal Inference Architectures},
  author={Alex},
  year={2025},
  note={Based on the Intuition Machine 12-mode framework}
}
```

## Acknowledgments

- Intuition Machine for the theoretical framework
- Charles Sanders Peirce for the foundational work on inference
- The Entropic Bridge project for QRNG methodology
