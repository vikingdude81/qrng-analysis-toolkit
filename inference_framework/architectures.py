"""
Inference Architectures: Core Module

Based on the Intuition Machine framework extending Peirce's triad into
three fundamental architectures of validity-production.

Each architecture achieves valid conclusions through fundamentally different mechanisms:
- Strange Attractor: Convergent quality (monadic, self-referential)
- Code Duality: Perfect coupling (dyadic, deterministic)
- Tensegrity: Balanced forces (tetradic, dynamic opposition)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable
import json


class Architecture(Enum):
    """The three fundamental inference architectures."""
    STRANGE_ATTRACTOR = auto()  # Convergence
    CODE_DUALITY = auto()       # Coupling
    TENSEGRITY = auto()         # Balance


class InferenceMode(Enum):
    """
    The 12 modes of inference, each governed by one of the three architectures.
    
    Strange Attractor modes (convergent):
        - ABDUCTION (1): "What would explain this?"
        - QUALITATIVE (9): "What do qualities intimate?"
        - EQUILIBRIUM (12): "Where does the process converge?"
    
    Code Duality modes (coupled):
        - DEDUCTION (3): "What must follow?"
        - SYNTACTIC (4): "How is it organized?"
        - DIAGRAMMATIC (5): "What does manipulation reveal?"
        - METONYMIC (8): "What does convention connect?"
        - TYPE_TOKEN (11): "What does instantiation transfer?"
    
    Tensegrity modes (balanced):
        - INDUCTION (2): "What pattern holds?"
        - ANALOGICAL (6): "What corresponds across domains?"
        - INDEXICAL (7): "What does this trace show?"
        - CONTRASTIVE (10): "What does difference reveal?"
    """
    # Strange Attractor
    ABDUCTION = 1
    QUALITATIVE = 9
    EQUILIBRIUM = 12
    
    # Code Duality
    DEDUCTION = 3
    SYNTACTIC = 4
    DIAGRAMMATIC = 5
    METONYMIC = 8
    TYPE_TOKEN = 11
    
    # Tensegrity
    INDUCTION = 2
    ANALOGICAL = 6
    INDEXICAL = 7
    CONTRASTIVE = 10


# Map modes to their governing architecture
MODE_TO_ARCHITECTURE: Dict[InferenceMode, Architecture] = {
    InferenceMode.ABDUCTION: Architecture.STRANGE_ATTRACTOR,
    InferenceMode.QUALITATIVE: Architecture.STRANGE_ATTRACTOR,
    InferenceMode.EQUILIBRIUM: Architecture.STRANGE_ATTRACTOR,
    
    InferenceMode.DEDUCTION: Architecture.CODE_DUALITY,
    InferenceMode.SYNTACTIC: Architecture.CODE_DUALITY,
    InferenceMode.DIAGRAMMATIC: Architecture.CODE_DUALITY,
    InferenceMode.METONYMIC: Architecture.CODE_DUALITY,
    InferenceMode.TYPE_TOKEN: Architecture.CODE_DUALITY,
    
    InferenceMode.INDUCTION: Architecture.TENSEGRITY,
    InferenceMode.ANALOGICAL: Architecture.TENSEGRITY,
    InferenceMode.INDEXICAL: Architecture.TENSEGRITY,
    InferenceMode.CONTRASTIVE: Architecture.TENSEGRITY,
}


@dataclass
class InferenceState:
    """Tracks the state of an inference process."""
    mode: InferenceMode
    iteration: int = 0
    converged: bool = False
    confidence: float = 0.0
    intermediate_results: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.name,
            "iteration": self.iteration,
            "converged": self.converged,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class InferenceResult:
    """The result of an inference process."""
    conclusion: Any
    state: InferenceState
    reasoning_trace: List[str] = field(default_factory=list)
    alternative_conclusions: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "conclusion": self.conclusion,
            "state": self.state.to_dict(),
            "reasoning_trace": self.reasoning_trace,
            "alternatives": self.alternative_conclusions
        }


class BaseArchitecture(ABC):
    """
    Abstract base class for inference architectures.
    
    Each architecture implements a fundamentally different process
    for achieving valid conclusions.
    """
    
    def __init__(self, randomness_source: Optional[Callable[[], float]] = None):
        """
        Initialize the architecture.
        
        Args:
            randomness_source: Optional function that returns random floats in [0,1).
                              If None, uses default PRNG. Can be swapped for QRNG
                              for entropy sensitivity experiments.
        """
        self.randomness_source = randomness_source or self._default_random
        self._random_calls = 0  # Track for experiments
    
    def _default_random(self) -> float:
        """Default PRNG source."""
        import random
        return random.random()
    
    def get_random(self) -> float:
        """Get a random value and track the call."""
        self._random_calls += 1
        return self.randomness_source()
    
    @property
    def random_call_count(self) -> int:
        """Number of random values consumed."""
        return self._random_calls
    
    def reset_random_counter(self):
        """Reset the random call counter."""
        self._random_calls = 0
    
    @property
    @abstractmethod
    def architecture_type(self) -> Architecture:
        """Return the architecture type."""
        pass
    
    @property
    @abstractmethod
    def supported_modes(self) -> List[InferenceMode]:
        """Return list of inference modes this architecture supports."""
        pass
    
    @abstractmethod
    def infer(
        self, 
        mode: InferenceMode,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        max_iterations: int = 10
    ) -> InferenceResult:
        """
        Execute inference in the specified mode.
        
        Args:
            mode: The specific inference mode to use
            inputs: Mode-specific inputs (observation, premises, etc.)
            llm_call: Function to call the LLM with a prompt
            max_iterations: Maximum iterations for iterative processes
            
        Returns:
            InferenceResult containing the conclusion and process metadata
        """
        pass


class StrangeAttractorArchitecture(BaseArchitecture):
    """
    The Architecture of Convergence.
    
    Strange Attractor inferences achieve stability through convergence
    toward a stable point or "basin of attraction." They are monadic
    and self-referential, operating on qualities and feelings to
    produce insight.
    
    Key characteristics:
    - Driven by "immediate attraction for the idea itself"
    - Process is iterative and clarifying
    - Conclusion is a stable "fixed point" of understanding
    
    Modes: Abduction (1), Qualitative (9), Equilibrium (12)
    """
    
    @property
    def architecture_type(self) -> Architecture:
        return Architecture.STRANGE_ATTRACTOR
    
    @property
    def supported_modes(self) -> List[InferenceMode]:
        return [
            InferenceMode.ABDUCTION,
            InferenceMode.QUALITATIVE,
            InferenceMode.EQUILIBRIUM
        ]
    
    def infer(
        self,
        mode: InferenceMode,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        max_iterations: int = 10
    ) -> InferenceResult:
        
        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} not supported by Strange Attractor architecture")
        
        state = InferenceState(mode=mode)
        reasoning_trace = []
        
        if mode == InferenceMode.ABDUCTION:
            return self._abductive_inference(inputs, llm_call, max_iterations, state, reasoning_trace)
        elif mode == InferenceMode.QUALITATIVE:
            return self._qualitative_inference(inputs, llm_call, max_iterations, state, reasoning_trace)
        elif mode == InferenceMode.EQUILIBRIUM:
            return self._equilibrium_inference(inputs, llm_call, max_iterations, state, reasoning_trace)
    
    def _abductive_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        max_iterations: int,
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 1: Abduction - "What would explain this surprising fact?"
        
        Iteratively generate and refine hypotheses until convergence
        on a stable explanation.
        """
        observation = inputs.get("observation", "")
        context = inputs.get("context", "")
        
        # Initial hypothesis generation with randomness for exploration
        temperature_boost = self.get_random() * 0.3  # Vary exploration
        
        prompt = f"""You are performing abductive inference - generating hypotheses to explain a surprising observation.

Observation: {observation}
Context: {context}

Generate an initial hypothesis that could explain this observation. Focus on finding the most plausible explanation.

Respond with your hypothesis and your confidence (0-1) in this format:
HYPOTHESIS: [your hypothesis]
CONFIDENCE: [0.0-1.0]
REASONING: [why this hypothesis]"""

        response = llm_call(prompt)
        current_hypothesis = self._parse_hypothesis(response)
        state.intermediate_results.append(current_hypothesis)
        reasoning_trace.append(f"Initial hypothesis: {current_hypothesis.get('hypothesis', 'unknown')}")
        
        previous_hypothesis = None
        alternatives = []
        
        for i in range(max_iterations):
            state.iteration = i + 1
            
            # Check for convergence (hypothesis stabilized)
            if previous_hypothesis and self._hypotheses_similar(current_hypothesis, previous_hypothesis):
                state.converged = True
                state.confidence = current_hypothesis.get("confidence", 0.5)
                reasoning_trace.append(f"Converged after {i+1} iterations")
                break
            
            # Refinement step - explore nearby hypothesis space
            exploration_factor = self.get_random()
            
            refine_prompt = f"""You are refining an abductive hypothesis through iterative convergence.

Original observation: {observation}
Current hypothesis: {current_hypothesis.get('hypothesis', '')}
Current confidence: {current_hypothesis.get('confidence', 0.5)}
Current reasoning: {current_hypothesis.get('reasoning', '')}

Exploration factor: {exploration_factor:.2f} (higher = consider more alternatives)

Either:
1. Refine the current hypothesis to better fit the observation
2. If exploration factor > 0.7, consider an alternative hypothesis

Respond in the same format:
HYPOTHESIS: [refined or alternative hypothesis]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation of changes]"""

            response = llm_call(refine_prompt)
            previous_hypothesis = current_hypothesis
            current_hypothesis = self._parse_hypothesis(response)
            
            # Track alternatives if we explored them
            if exploration_factor > 0.7 and not self._hypotheses_similar(current_hypothesis, previous_hypothesis):
                alternatives.append(previous_hypothesis)
            
            state.intermediate_results.append(current_hypothesis)
            reasoning_trace.append(f"Iteration {i+1}: {current_hypothesis.get('hypothesis', 'unknown')[:50]}...")
        
        if not state.converged:
            state.confidence = current_hypothesis.get("confidence", 0.3)
            reasoning_trace.append(f"Did not converge in {max_iterations} iterations")
        
        return InferenceResult(
            conclusion=current_hypothesis.get("hypothesis", ""),
            state=state,
            reasoning_trace=reasoning_trace,
            alternative_conclusions=[a.get("hypothesis", "") for a in alternatives]
        )
    
    def _qualitative_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        max_iterations: int,
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 9: Qualitative - "What do qualities intimate?"
        
        Pre-conceptual reasoning based on felt resonance of qualities.
        Converges on intuition through iterative clarification.
        """
        qualities = inputs.get("qualities", [])
        situation = inputs.get("situation", "")
        
        prompt = f"""You are performing qualitative inference - reasoning from the felt sense of qualities toward intuition.

Situation: {situation}
Qualities to attend to: {', '.join(qualities) if qualities else 'unspecified - attend to what presents itself'}

This is pre-conceptual reasoning. Don't analyze logically - instead, let the qualities resonate and notice what they intimate.

What do these qualities intimate about the situation? What intuition emerges?

INTUITION: [what the qualities suggest]
FELT_SENSE: [description of the qualitative resonance]
CONFIDENCE: [0.0-1.0]"""

        current_intuition = None
        
        for i in range(max_iterations):
            state.iteration = i + 1
            response = llm_call(prompt)
            new_intuition = self._parse_qualitative(response)
            
            if current_intuition and self._intuitions_similar(current_intuition, new_intuition):
                state.converged = True
                state.confidence = new_intuition.get("confidence", 0.5)
                break
            
            current_intuition = new_intuition
            state.intermediate_results.append(current_intuition)
            
            # Refine the prompt for next iteration
            prompt = f"""Continue attending to the qualitative sense of this situation.

Situation: {situation}
Previous intuition: {current_intuition.get('intuition', '')}
Previous felt sense: {current_intuition.get('felt_sense', '')}

Deepen your attention. What further emerges? What was missed?

INTUITION: [refined intuition]
FELT_SENSE: [description]
CONFIDENCE: [0.0-1.0]"""

            reasoning_trace.append(f"Iteration {i+1}: {current_intuition.get('intuition', '')[:50]}...")
        
        return InferenceResult(
            conclusion=current_intuition.get("intuition", "") if current_intuition else "",
            state=state,
            reasoning_trace=reasoning_trace
        )
    
    def _equilibrium_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        max_iterations: int,
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 12: Equilibrium - "Where does this entire process converge?"
        
        Meta-level inference about the stable endpoints of a dynamic system.
        """
        system = inputs.get("system", "")
        dynamics = inputs.get("dynamics", "")
        initial_conditions = inputs.get("initial_conditions", "")
        
        prompt = f"""You are performing equilibrium inference - determining where a dynamic process converges.

System: {system}
Dynamics: {dynamics}
Initial conditions: {initial_conditions}

This is meta-level reasoning about fixed points and attractors. Where does this system stabilize?

EQUILIBRIUM: [description of the stable state(s)]
STABILITY: [stable/unstable/limit_cycle/chaotic]
PATH: [description of how the system reaches equilibrium]
CONFIDENCE: [0.0-1.0]"""

        response = llm_call(prompt)
        result = self._parse_equilibrium(response)
        
        state.iteration = 1
        state.converged = True  # Equilibrium inference is typically one-shot
        state.confidence = result.get("confidence", 0.5)
        
        return InferenceResult(
            conclusion=result.get("equilibrium", ""),
            state=state,
            reasoning_trace=[f"Equilibrium analysis: {result.get('stability', 'unknown')} state"]
        )
    
    def _parse_hypothesis(self, response: str) -> Dict[str, Any]:
        """Parse hypothesis response format."""
        result = {"hypothesis": "", "confidence": 0.5, "reasoning": ""}
        for line in response.split("\n"):
            if line.startswith("HYPOTHESIS:"):
                result["hypothesis"] = line.replace("HYPOTHESIS:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
        return result
    
    def _parse_qualitative(self, response: str) -> Dict[str, Any]:
        """Parse qualitative inference response."""
        result = {"intuition": "", "felt_sense": "", "confidence": 0.5}
        for line in response.split("\n"):
            if line.startswith("INTUITION:"):
                result["intuition"] = line.replace("INTUITION:", "").strip()
            elif line.startswith("FELT_SENSE:"):
                result["felt_sense"] = line.replace("FELT_SENSE:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
        return result
    
    def _parse_equilibrium(self, response: str) -> Dict[str, Any]:
        """Parse equilibrium inference response."""
        result = {"equilibrium": "", "stability": "", "path": "", "confidence": 0.5}
        for line in response.split("\n"):
            if line.startswith("EQUILIBRIUM:"):
                result["equilibrium"] = line.replace("EQUILIBRIUM:", "").strip()
            elif line.startswith("STABILITY:"):
                result["stability"] = line.replace("STABILITY:", "").strip()
            elif line.startswith("PATH:"):
                result["path"] = line.replace("PATH:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
        return result
    
    def _hypotheses_similar(self, h1: Dict, h2: Dict, threshold: float = 0.05) -> bool:
        """Check if two hypotheses are similar enough to indicate convergence."""
        # Simple heuristic: confidence delta small and same core concept
        conf_diff = abs(h1.get("confidence", 0) - h2.get("confidence", 0))
        # In practice, would use semantic similarity
        return conf_diff < threshold and h1.get("hypothesis", "")[:50] == h2.get("hypothesis", "")[:50]
    
    def _intuitions_similar(self, i1: Dict, i2: Dict) -> bool:
        """Check if two intuitions are similar."""
        return i1.get("intuition", "")[:50] == i2.get("intuition", "")[:50]


class CodeDualityArchitecture(BaseArchitecture):
    """
    The Architecture of Coupling.
    
    Code-Dual inferences are defined by perfect, mutual correspondence
    between two distinct domains (e.g., proof and program, type and token).
    They are dyadic, achieving certainty through unbreakable coupling.
    
    Key characteristics:
    - Based on mutual, deterministic mapping
    - Validity is absolute within the system
    - Represents the "necessary inference" aspect of reason
    
    Modes: Deduction (3), Syntactic (4), Diagrammatic (5), Metonymic (8), Type-Token (11)
    """
    
    @property
    def architecture_type(self) -> Architecture:
        return Architecture.CODE_DUALITY
    
    @property
    def supported_modes(self) -> List[InferenceMode]:
        return [
            InferenceMode.DEDUCTION,
            InferenceMode.SYNTACTIC,
            InferenceMode.DIAGRAMMATIC,
            InferenceMode.METONYMIC,
            InferenceMode.TYPE_TOKEN
        ]
    
    def infer(
        self,
        mode: InferenceMode,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        max_iterations: int = 1  # Code Duality is typically deterministic
    ) -> InferenceResult:
        
        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} not supported by Code Duality architecture")
        
        state = InferenceState(mode=mode, iteration=1)
        reasoning_trace = []
        
        # Code Duality inferences are typically single-step (deterministic)
        # Randomness only enters in rule selection, not in rule application
        
        if mode == InferenceMode.DEDUCTION:
            return self._deductive_inference(inputs, llm_call, state, reasoning_trace)
        elif mode == InferenceMode.SYNTACTIC:
            return self._syntactic_inference(inputs, llm_call, state, reasoning_trace)
        elif mode == InferenceMode.DIAGRAMMATIC:
            return self._diagrammatic_inference(inputs, llm_call, state, reasoning_trace)
        elif mode == InferenceMode.METONYMIC:
            return self._metonymic_inference(inputs, llm_call, state, reasoning_trace)
        elif mode == InferenceMode.TYPE_TOKEN:
            return self._type_token_inference(inputs, llm_call, state, reasoning_trace)
    
    def _deductive_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 3: Deduction - "What must follow from these premises?"
        
        Necessary inference where premises and conclusions are
        perfectly coupled by rules.
        """
        premises = inputs.get("premises", [])
        rules = inputs.get("rules", [])
        
        prompt = f"""You are performing deductive inference - deriving necessary conclusions from premises.

Premises:
{chr(10).join(f'  {i+1}. {p}' for i, p in enumerate(premises))}

Rules of inference:
{chr(10).join(f'  - {r}' for r in rules) if rules else '  - Use standard logical rules'}

What conclusion(s) MUST follow from these premises? This is necessary inference - only include conclusions that are guaranteed by the premises.

CONCLUSION: [what must follow]
DERIVATION: [step-by-step derivation]
CERTAINTY: [certain/probable/uncertain]"""

        response = llm_call(prompt)
        result = self._parse_deduction(response)
        
        state.converged = True
        state.confidence = 1.0 if result.get("certainty") == "certain" else 0.7
        reasoning_trace.append(f"Deduction: {result.get('derivation', '')[:100]}")
        
        return InferenceResult(
            conclusion=result.get("conclusion", ""),
            state=state,
            reasoning_trace=reasoning_trace
        )
    
    def _syntactic_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 4: Syntactic - "How is this system organized?"
        
        Reasoning based on organizational structure and rules,
        independent of meaning or content.
        """
        structure = inputs.get("structure", "")
        system_rules = inputs.get("system_rules", "")
        
        prompt = f"""You are performing syntactic inference - analyzing organizational structure independent of meaning.

Structure to analyze: {structure}
System rules: {system_rules if system_rules else 'Infer from the structure'}

Analyze the organizational structure. What is the grammar, logical form, or architecture?

ORGANIZATION: [structural analysis]
RULES_IDENTIFIED: [rules governing the structure]
VALIDITY: [does it conform to its own rules?]"""

        response = llm_call(prompt)
        result = self._parse_syntactic(response)
        
        state.converged = True
        state.confidence = 0.9
        
        return InferenceResult(
            conclusion=result.get("organization", ""),
            state=state,
            reasoning_trace=[f"Syntactic analysis: {result.get('rules_identified', '')}"]
        )
    
    def _diagrammatic_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 5: Diagrammatic - "What does manipulation of this representation reveal?"
        
        Discovering truths by manipulating diagrams according to rules.
        The diagram and rules form a code-dual pair.
        """
        diagram = inputs.get("diagram", "")
        manipulation_rules = inputs.get("rules", "")
        goal = inputs.get("goal", "")
        
        prompt = f"""You are performing diagrammatic inference - discovering truths through diagram manipulation.

Diagram/representation: {diagram}
Manipulation rules: {manipulation_rules}
Goal: {goal if goal else 'Discover what the diagram reveals'}

Apply the rules to manipulate the diagram. What necessary conclusions emerge from the manipulation?

MANIPULATION_STEPS: [sequence of rule applications]
REVELATION: [what the manipulation reveals]
NECESSITY: [why this conclusion is necessary given the rules]"""

        response = llm_call(prompt)
        result = self._parse_diagrammatic(response)
        
        state.converged = True
        state.confidence = 0.95
        
        return InferenceResult(
            conclusion=result.get("revelation", ""),
            state=state,
            reasoning_trace=[f"Diagrammatic manipulation: {result.get('manipulation_steps', '')}"]
        )
    
    def _metonymic_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 8: Metonymic - "What does convention associate with this?"
        
        Reasoning through established contiguity or convention.
        Word and referent are linked in a code-dual of cultural habit.
        """
        term = inputs.get("term", "")
        context = inputs.get("context", "")
        
        prompt = f"""You are performing metonymic inference - reasoning through conventional association.

Term/symbol: {term}
Context: {context}

What does convention or cultural habit associate with this term? (e.g., "The Crown" → the monarchy)

ASSOCIATION: [what the term conventionally refers to]
BASIS: [the conventional/cultural link]
STRENGTH: [strong/moderate/weak convention]"""

        response = llm_call(prompt)
        result = self._parse_metonymic(response)
        
        state.converged = True
        state.confidence = 0.85
        
        return InferenceResult(
            conclusion=result.get("association", ""),
            state=state,
            reasoning_trace=[f"Metonymic link: {result.get('basis', '')}"]
        )
    
    def _type_token_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 11: Type-Token - "What does instantiation transfer from type to token?"
        
        Transferring properties from a general type to a specific instance.
        """
        type_description = inputs.get("type", "")
        token = inputs.get("token", "")
        
        prompt = f"""You are performing type-token inference - determining what properties transfer from type to instance.

Type (general): {type_description}
Token (specific instance): {token}

What properties of the type necessarily apply to this token? What is modified in instantiation?

TRANSFERRED_PROPERTIES: [properties that apply to the token]
MODIFIED_PROPERTIES: [properties changed by instantiation]
TOKEN_SPECIFIC: [properties unique to this token]"""

        response = llm_call(prompt)
        result = self._parse_type_token(response)
        
        state.converged = True
        state.confidence = 0.9
        
        return InferenceResult(
            conclusion=result.get("transferred_properties", ""),
            state=state,
            reasoning_trace=[f"Type→Token: {result.get('modified_properties', '')}"]
        )
    
    def _parse_deduction(self, response: str) -> Dict[str, Any]:
        result = {"conclusion": "", "derivation": "", "certainty": "uncertain"}
        for line in response.split("\n"):
            if line.startswith("CONCLUSION:"):
                result["conclusion"] = line.replace("CONCLUSION:", "").strip()
            elif line.startswith("DERIVATION:"):
                result["derivation"] = line.replace("DERIVATION:", "").strip()
            elif line.startswith("CERTAINTY:"):
                result["certainty"] = line.replace("CERTAINTY:", "").strip().lower()
        return result
    
    def _parse_syntactic(self, response: str) -> Dict[str, Any]:
        result = {"organization": "", "rules_identified": "", "validity": ""}
        for line in response.split("\n"):
            if line.startswith("ORGANIZATION:"):
                result["organization"] = line.replace("ORGANIZATION:", "").strip()
            elif line.startswith("RULES_IDENTIFIED:"):
                result["rules_identified"] = line.replace("RULES_IDENTIFIED:", "").strip()
            elif line.startswith("VALIDITY:"):
                result["validity"] = line.replace("VALIDITY:", "").strip()
        return result
    
    def _parse_diagrammatic(self, response: str) -> Dict[str, Any]:
        result = {"manipulation_steps": "", "revelation": "", "necessity": ""}
        for line in response.split("\n"):
            if line.startswith("MANIPULATION_STEPS:"):
                result["manipulation_steps"] = line.replace("MANIPULATION_STEPS:", "").strip()
            elif line.startswith("REVELATION:"):
                result["revelation"] = line.replace("REVELATION:", "").strip()
            elif line.startswith("NECESSITY:"):
                result["necessity"] = line.replace("NECESSITY:", "").strip()
        return result
    
    def _parse_metonymic(self, response: str) -> Dict[str, Any]:
        result = {"association": "", "basis": "", "strength": ""}
        for line in response.split("\n"):
            if line.startswith("ASSOCIATION:"):
                result["association"] = line.replace("ASSOCIATION:", "").strip()
            elif line.startswith("BASIS:"):
                result["basis"] = line.replace("BASIS:", "").strip()
            elif line.startswith("STRENGTH:"):
                result["strength"] = line.replace("STRENGTH:", "").strip()
        return result
    
    def _parse_type_token(self, response: str) -> Dict[str, Any]:
        result = {"transferred_properties": "", "modified_properties": "", "token_specific": ""}
        for line in response.split("\n"):
            if line.startswith("TRANSFERRED_PROPERTIES:"):
                result["transferred_properties"] = line.replace("TRANSFERRED_PROPERTIES:", "").strip()
            elif line.startswith("MODIFIED_PROPERTIES:"):
                result["modified_properties"] = line.replace("MODIFIED_PROPERTIES:", "").strip()
            elif line.startswith("TOKEN_SPECIFIC:"):
                result["token_specific"] = line.replace("TOKEN_SPECIFIC:", "").strip()
        return result


class TensegrityArchitecture(BaseArchitecture):
    """
    The Architecture of Balance.
    
    Tensegrity inferences achieve stability not through convergence or
    coupling, but through a dynamic balance of opposing forces. They are
    tetradic, requiring at least four elements held in productive tension.
    
    Key characteristics:
    - Maintains stability through balanced opposition
    - The structure itself is the source of validity
    - Represents the "reasonableness" aspect of inference
    
    Modes: Induction (2), Analogical (6), Indexical (7), Contrastive (10)
    """
    
    @property
    def architecture_type(self) -> Architecture:
        return Architecture.TENSEGRITY
    
    @property
    def supported_modes(self) -> List[InferenceMode]:
        return [
            InferenceMode.INDUCTION,
            InferenceMode.ANALOGICAL,
            InferenceMode.INDEXICAL,
            InferenceMode.CONTRASTIVE
        ]
    
    def infer(
        self,
        mode: InferenceMode,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        max_iterations: int = 5
    ) -> InferenceResult:
        
        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} not supported by Tensegrity architecture")
        
        state = InferenceState(mode=mode)
        reasoning_trace = []
        
        if mode == InferenceMode.INDUCTION:
            return self._inductive_inference(inputs, llm_call, max_iterations, state, reasoning_trace)
        elif mode == InferenceMode.ANALOGICAL:
            return self._analogical_inference(inputs, llm_call, state, reasoning_trace)
        elif mode == InferenceMode.INDEXICAL:
            return self._indexical_inference(inputs, llm_call, state, reasoning_trace)
        elif mode == InferenceMode.CONTRASTIVE:
            return self._contrastive_inference(inputs, llm_call, state, reasoning_trace)
    
    def _inductive_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        max_iterations: int,
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 2: Induction - "What pattern holds across these cases?"
        
        Generalizing a pattern requires balancing confirming instances
        against potential counter-examples, holding evidence and theory
        in productive tension.
        """
        instances = inputs.get("instances", [])
        domain = inputs.get("domain", "")
        
        # Tetradic structure: instances, pattern, counter-examples, confidence
        prompt = f"""You are performing inductive inference - finding patterns while accounting for exceptions.

Domain: {domain}
Instances observed:
{chr(10).join(f'  - {inst}' for inst in instances)}

This is tensegrity inference: you must balance multiple forces:
1. The pattern suggested by instances
2. Potential counter-examples
3. The scope/boundary of the generalization
4. Confidence given the evidence

What pattern emerges? Hold these elements in tension.

PATTERN: [the generalized pattern]
CONFIRMING_EVIDENCE: [what supports this pattern]
POTENTIAL_COUNTEREXAMPLES: [what might falsify it]
SCOPE: [where does this pattern apply?]
CONFIDENCE: [0.0-1.0]"""

        response = llm_call(prompt)
        result = self._parse_induction(response)
        
        # Tensegrity: iterate to find stable balance
        tensions = self._compute_tensions(result)
        
        for i in range(max_iterations):
            state.iteration = i + 1
            
            if self._is_balanced(tensions):
                state.converged = True
                break
            
            # Adjust based on imbalance - sample which force to strengthen
            force_to_adjust = self._sample_force(tensions)
            
            adjust_prompt = f"""The inductive inference is not yet balanced.

Current pattern: {result.get('pattern', '')}
Tensions: {tensions}
Force to strengthen: {force_to_adjust}

Refine the inference to better balance all four elements.

PATTERN: [refined pattern]
CONFIRMING_EVIDENCE: [updated evidence]
POTENTIAL_COUNTEREXAMPLES: [updated counter-examples]
SCOPE: [refined scope]
CONFIDENCE: [0.0-1.0]"""

            response = llm_call(adjust_prompt)
            result = self._parse_induction(response)
            tensions = self._compute_tensions(result)
            reasoning_trace.append(f"Iteration {i+1}: rebalancing {force_to_adjust}")
        
        state.confidence = result.get("confidence", 0.5)
        
        return InferenceResult(
            conclusion=result.get("pattern", ""),
            state=state,
            reasoning_trace=reasoning_trace
        )
    
    def _analogical_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 6: Analogical - "What corresponds between these domains?"
        
        Transfer relational structure from source to target domain.
        Source and target are held in tension by identifying similarities
        while respecting differences.
        """
        source_domain = inputs.get("source", {})
        target_domain = inputs.get("target", {})
        
        prompt = f"""You are performing analogical inference - mapping structure between domains.

Source domain: {source_domain}
Target domain: {target_domain}

This is tensegrity inference: hold source and target in productive tension.
- What structural correspondences exist?
- What differences must be respected?
- What can be transferred? What cannot?

CORRESPONDENCES: [structural mappings from source to target]
DIFFERENCES: [where the analogy breaks down]
TRANSFERRED_INSIGHT: [what the source illuminates about the target]
LIMITATIONS: [where the analogy should not be pushed]
CONFIDENCE: [0.0-1.0]"""

        response = llm_call(prompt)
        result = self._parse_analogical(response)
        
        state.iteration = 1
        state.converged = True
        state.confidence = result.get("confidence", 0.6)
        
        reasoning_trace.append(f"Analogical mapping: {result.get('correspondences', '')[:50]}...")
        
        return InferenceResult(
            conclusion=result.get("transferred_insight", ""),
            state=state,
            reasoning_trace=reasoning_trace,
            alternative_conclusions=[result.get("limitations", "")]
        )
    
    def _indexical_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 7: Indexical - "What does this trace/sign show about its cause?"
        
        Causal tracking from effect back to cause. Tensegrity between:
        the present sign, the absent cause, the interpretive law, and the observer.
        """
        sign = inputs.get("sign", "")
        context = inputs.get("context", "")
        
        prompt = f"""You are performing indexical inference - tracking from effect to cause.

Sign/trace observed: {sign}
Context: {context}

This is tensegrity inference with four elements in tension:
1. The present sign (what you observe)
2. The absent cause (what produced it)
3. The interpretive law (how signs relate to causes)
4. The observer's position (how context affects interpretation)

What cause does this sign indicate?

SIGN_DESCRIPTION: [what is actually observed]
INFERRED_CAUSE: [what produced this sign]
INTERPRETIVE_LAW: [the connection between this type of sign and cause]
OBSERVER_EFFECTS: [how context/position affects the inference]
CONFIDENCE: [0.0-1.0]"""

        response = llm_call(prompt)
        result = self._parse_indexical(response)
        
        state.iteration = 1
        state.converged = True
        state.confidence = result.get("confidence", 0.6)
        
        return InferenceResult(
            conclusion=result.get("inferred_cause", ""),
            state=state,
            reasoning_trace=[f"Indexical: {result.get('interpretive_law', '')}"]
        )
    
    def _contrastive_inference(
        self,
        inputs: Dict[str, Any],
        llm_call: Callable[[str], str],
        state: InferenceState,
        reasoning_trace: List[str]
    ) -> InferenceResult:
        """
        Mode 10: Contrastive - "What does the difference between these reveal?"
        
        Reasoning by placing two concepts in opposition. Each is stabilized
        and illuminated by its contrast with the other.
        """
        concept_a = inputs.get("concept_a", "")
        concept_b = inputs.get("concept_b", "")
        
        prompt = f"""You are performing contrastive inference - reasoning through opposition.

Concept A: {concept_a}
Concept B: {concept_b}

Place these concepts in productive opposition. What does their difference reveal?
The meaning of each is stabilized by its contrast with the other.

CONTRAST: [the key difference between A and B]
INSIGHT_ABOUT_A: [what the contrast reveals about A]
INSIGHT_ABOUT_B: [what the contrast reveals about B]
SYNTHESIZED_UNDERSTANDING: [what we understand through the opposition]
CONFIDENCE: [0.0-1.0]"""

        response = llm_call(prompt)
        result = self._parse_contrastive(response)
        
        state.iteration = 1
        state.converged = True
        state.confidence = result.get("confidence", 0.7)
        
        return InferenceResult(
            conclusion=result.get("synthesized_understanding", ""),
            state=state,
            reasoning_trace=[f"Contrast: {result.get('contrast', '')}"]
        )
    
    def _compute_tensions(self, result: Dict) -> Dict[str, float]:
        """Compute tension values between the four elements."""
        # Simplified: in practice, would use semantic analysis
        return {
            "pattern_evidence": 0.5,
            "evidence_counter": 0.5,
            "counter_scope": 0.5,
            "scope_pattern": 0.5
        }
    
    def _is_balanced(self, tensions: Dict[str, float], threshold: float = 0.1) -> bool:
        """Check if tensions are in balance."""
        values = list(tensions.values())
        return max(values) - min(values) < threshold
    
    def _sample_force(self, tensions: Dict[str, float]) -> str:
        """Sample which force to strengthen based on randomness."""
        # Use randomness to explore different rebalancing strategies
        r = self.get_random()
        forces = list(tensions.keys())
        return forces[int(r * len(forces))]
    
    def _parse_induction(self, response: str) -> Dict[str, Any]:
        result = {"pattern": "", "confirming_evidence": "", "potential_counterexamples": "", "scope": "", "confidence": 0.5}
        for line in response.split("\n"):
            if line.startswith("PATTERN:"):
                result["pattern"] = line.replace("PATTERN:", "").strip()
            elif line.startswith("CONFIRMING_EVIDENCE:"):
                result["confirming_evidence"] = line.replace("CONFIRMING_EVIDENCE:", "").strip()
            elif line.startswith("POTENTIAL_COUNTEREXAMPLES:"):
                result["potential_counterexamples"] = line.replace("POTENTIAL_COUNTEREXAMPLES:", "").strip()
            elif line.startswith("SCOPE:"):
                result["scope"] = line.replace("SCOPE:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
        return result
    
    def _parse_analogical(self, response: str) -> Dict[str, Any]:
        result = {"correspondences": "", "differences": "", "transferred_insight": "", "limitations": "", "confidence": 0.5}
        for line in response.split("\n"):
            if line.startswith("CORRESPONDENCES:"):
                result["correspondences"] = line.replace("CORRESPONDENCES:", "").strip()
            elif line.startswith("DIFFERENCES:"):
                result["differences"] = line.replace("DIFFERENCES:", "").strip()
            elif line.startswith("TRANSFERRED_INSIGHT:"):
                result["transferred_insight"] = line.replace("TRANSFERRED_INSIGHT:", "").strip()
            elif line.startswith("LIMITATIONS:"):
                result["limitations"] = line.replace("LIMITATIONS:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
        return result
    
    def _parse_indexical(self, response: str) -> Dict[str, Any]:
        result = {"sign_description": "", "inferred_cause": "", "interpretive_law": "", "observer_effects": "", "confidence": 0.5}
        for line in response.split("\n"):
            if line.startswith("SIGN_DESCRIPTION:"):
                result["sign_description"] = line.replace("SIGN_DESCRIPTION:", "").strip()
            elif line.startswith("INFERRED_CAUSE:"):
                result["inferred_cause"] = line.replace("INFERRED_CAUSE:", "").strip()
            elif line.startswith("INTERPRETIVE_LAW:"):
                result["interpretive_law"] = line.replace("INTERPRETIVE_LAW:", "").strip()
            elif line.startswith("OBSERVER_EFFECTS:"):
                result["observer_effects"] = line.replace("OBSERVER_EFFECTS:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
        return result
    
    def _parse_contrastive(self, response: str) -> Dict[str, Any]:
        result = {"contrast": "", "insight_about_a": "", "insight_about_b": "", "synthesized_understanding": "", "confidence": 0.5}
        for line in response.split("\n"):
            if line.startswith("CONTRAST:"):
                result["contrast"] = line.replace("CONTRAST:", "").strip()
            elif line.startswith("INSIGHT_ABOUT_A:"):
                result["insight_about_a"] = line.replace("INSIGHT_ABOUT_A:", "").strip()
            elif line.startswith("INSIGHT_ABOUT_B:"):
                result["insight_about_b"] = line.replace("INSIGHT_ABOUT_B:", "").strip()
            elif line.startswith("SYNTHESIZED_UNDERSTANDING:"):
                result["synthesized_understanding"] = line.replace("SYNTHESIZED_UNDERSTANDING:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
        return result
