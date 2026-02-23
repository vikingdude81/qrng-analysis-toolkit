"""
Mode Classifier: Identifies which inference mode a reasoning task requires.

Uses pattern matching on the structure of the question/task to route
to the appropriate inference architecture and mode.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import re

from .architectures import (
    Architecture,
    InferenceMode,
    MODE_TO_ARCHITECTURE,
    StrangeAttractorArchitecture,
    CodeDualityArchitecture,
    TensegrityArchitecture,
    BaseArchitecture
)


@dataclass
class ClassificationResult:
    """Result of classifying a reasoning task."""
    mode: InferenceMode
    architecture: Architecture
    confidence: float
    signals: List[str]  # What triggered this classification
    alternative_modes: List[Tuple[InferenceMode, float]]  # Other possible modes


class ModeClassifier:
    """
    Classifies reasoning tasks into one of the 12 inference modes.
    
    Uses a combination of:
    1. Keyword/pattern matching for fast classification
    2. Structural analysis of the question type
    3. Optional LLM-based classification for ambiguous cases
    """
    
    # Pattern definitions for each mode
    MODE_PATTERNS: Dict[InferenceMode, Dict[str, Any]] = {
        # Strange Attractor modes
        InferenceMode.ABDUCTION: {
            "question_patterns": [
                r"what (would|could|might) explain",
                r"why (did|does|would|might)",
                r"what (caused|led to|resulted in)",
                r"hypothesis for",
                r"best explanation",
                r"account for (this|these|the)",
            ],
            "signals": ["surprising fact", "explanation needed", "hypothesis generation"],
            "requires": ["observation", "anomaly", "surprise"],
        },
        InferenceMode.QUALITATIVE: {
            "question_patterns": [
                r"what (do|does) .* (feel|seem) like",
                r"what('s| is) the (vibe|feel|sense|quality)",
                r"(intuition|gut feeling) about",
                r"what (qualities|characteristics) intimate",
                r"resonat(e|es|ing)",
            ],
            "signals": ["felt sense", "qualities", "intuition", "pre-conceptual"],
            "requires": ["qualities", "felt sense"],
        },
        InferenceMode.EQUILIBRIUM: {
            "question_patterns": [
                r"where (does|will) .* (converge|stabilize|settle)",
                r"(equilibrium|stable state|fixed point)",
                r"(long-term|eventual|final) (state|outcome|result)",
                r"what (does|will) .* (become|turn into)",
                r"dynamic.*converge",
            ],
            "signals": ["convergence", "stability", "final state", "dynamics"],
            "requires": ["system", "dynamics"],
        },
        
        # Code Duality modes
        InferenceMode.DEDUCTION: {
            "question_patterns": [
                r"what (must|necessarily) follow",
                r"(given|assuming) .* (therefore|then|conclude)",
                r"if .* then what",
                r"(prove|derive|demonstrate) that",
                r"(logical|necessary) (conclusion|consequence)",
            ],
            "signals": ["premises given", "necessary conclusion", "logical derivation"],
            "requires": ["premises", "rules"],
        },
        InferenceMode.SYNTACTIC: {
            "question_patterns": [
                r"how is .* (organized|structured)",
                r"what('s| is) the (structure|grammar|syntax|form)",
                r"(parse|analyze) the structure",
                r"(organizational|structural) (pattern|rules)",
                r"independent of (meaning|content|semantics)",
            ],
            "signals": ["structure", "organization", "syntax", "form over content"],
            "requires": ["structure", "system"],
        },
        InferenceMode.DIAGRAMMATIC: {
            "question_patterns": [
                r"(manipulate|transform) .* (diagram|representation)",
                r"what (does|would) .* (reveal|show) (when|if) (manipulated|transformed)",
                r"(geometric|diagrammatic) (proof|reasoning)",
                r"(construct|draw|modify) .* to (show|prove|demonstrate)",
            ],
            "signals": ["diagram manipulation", "visual reasoning", "construction"],
            "requires": ["diagram", "transformation rules"],
        },
        InferenceMode.METONYMIC: {
            "question_patterns": [
                r"what (does|do) .* (stand for|represent|symbolize)",
                r"(conventional|cultural) (association|meaning)",
                r"(metonym|synecdoche|figure)",
                r"what (does|do) .* (conventionally|traditionally) (mean|refer to)",
            ],
            "signals": ["convention", "cultural association", "symbolic reference"],
            "requires": ["term", "context"],
        },
        InferenceMode.TYPE_TOKEN: {
            "question_patterns": [
                r"(instance|instantiation) of",
                r"(specific|particular) (example|case) of",
                r"(apply|transfer) .* (general|type) to (specific|token)",
                r"this .* (is a|as a) (type|kind) of",
                r"(inherit|inherits) .* from",
            ],
            "signals": ["type-instance relationship", "inheritance", "instantiation"],
            "requires": ["type", "token"],
        },
        
        # Tensegrity modes
        InferenceMode.INDUCTION: {
            "question_patterns": [
                r"what pattern (holds|emerges)",
                r"(generalize|generalizing) from",
                r"(based on|from) these (examples|cases|instances)",
                r"(infer|induce) .* (rule|pattern|principle)",
                r"(common|shared) (pattern|feature|property)",
            ],
            "signals": ["multiple instances", "pattern finding", "generalization"],
            "requires": ["instances", "cases"],
        },
        InferenceMode.ANALOGICAL: {
            "question_patterns": [
                r"(analogous|similar) to",
                r"(like|as|resembles) .* in (what way|how)",
                r"(map|mapping|transfer) .* (from|between) .* (to|and)",
                r"what (corresponds|correspond) (between|across)",
                r"(source|target) domain",
            ],
            "signals": ["two domains", "structural correspondence", "mapping"],
            "requires": ["source domain", "target domain"],
        },
        InferenceMode.INDEXICAL: {
            "question_patterns": [
                r"what (caused|produced) this",
                r"(trace|track|sign) .* (back|points) to",
                r"(evidence|indicator|symptom) of",
                r"(infer|deduce) .* (from|based on) .* (sign|trace|effect)",
                r"smoke .* fire",  # Classic example
            ],
            "signals": ["effect→cause", "sign interpretation", "causal tracking"],
            "requires": ["sign", "trace", "effect"],
        },
        InferenceMode.CONTRASTIVE: {
            "question_patterns": [
                r"(difference|contrast) between",
                r"(compare|comparing) .* (with|to|and)",
                r"(distinguish|distinguishing|differentiate)",
                r"why .* (rather than|instead of|not)",
                r"what (makes|separates) .* (different|distinct)",
            ],
            "signals": ["two concepts", "opposition", "difference"],
            "requires": ["concept A", "concept B"],
        },
    }
    
    def __init__(self, llm_call: Optional[Callable[[str], str]] = None):
        """
        Initialize the classifier.
        
        Args:
            llm_call: Optional LLM function for ambiguous cases
        """
        self.llm_call = llm_call
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self.compiled_patterns: Dict[InferenceMode, List[re.Pattern]] = {}
        for mode, config in self.MODE_PATTERNS.items():
            self.compiled_patterns[mode] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in config["question_patterns"]
            ]
    
    def classify(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None,
        use_llm: bool = False
    ) -> ClassificationResult:
        """
        Classify a reasoning task into an inference mode.
        
        Args:
            task: The task or question to classify
            context: Optional additional context about the task
            use_llm: Whether to use LLM for ambiguous cases
            
        Returns:
            ClassificationResult with the identified mode and confidence
        """
        context = context or {}
        
        # Score each mode based on pattern matching
        scores: Dict[InferenceMode, Tuple[float, List[str]]] = {}
        
        for mode, patterns in self.compiled_patterns.items():
            score, signals = self._score_mode(task, mode, patterns, context)
            scores[mode] = (score, signals)
        
        # Sort by score
        sorted_modes = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        
        top_mode, (top_score, top_signals) = sorted_modes[0]
        
        # If ambiguous and LLM available, use it
        if use_llm and self.llm_call and top_score < 0.5:
            return self._llm_classify(task, context, sorted_modes[:3])
        
        # Build alternative modes list
        alternatives = [
            (mode, score) 
            for mode, (score, _) in sorted_modes[1:4] 
            if score > 0.1
        ]
        
        return ClassificationResult(
            mode=top_mode,
            architecture=MODE_TO_ARCHITECTURE[top_mode],
            confidence=min(top_score, 1.0),
            signals=top_signals,
            alternative_modes=alternatives
        )
    
    def _score_mode(
        self, 
        task: str, 
        mode: InferenceMode,
        patterns: List[re.Pattern],
        context: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Score how well a task matches a given mode."""
        score = 0.0
        signals = []
        
        # Pattern matching
        for pattern in patterns:
            if pattern.search(task):
                score += 0.3
                signals.append(f"Pattern: {pattern.pattern[:30]}...")
        
        # Check for required elements in context
        config = self.MODE_PATTERNS[mode]
        for required in config.get("requires", []):
            if required in context:
                score += 0.2
                signals.append(f"Has required: {required}")
        
        # Check for signal words
        task_lower = task.lower()
        for signal in config.get("signals", []):
            if signal.lower() in task_lower:
                score += 0.15
                signals.append(f"Signal: {signal}")
        
        return score, signals
    
    def _llm_classify(
        self, 
        task: str, 
        context: Dict[str, Any],
        top_candidates: List[Tuple[InferenceMode, Tuple[float, List[str]]]]
    ) -> ClassificationResult:
        """Use LLM for ambiguous classification."""
        
        mode_descriptions = {
            InferenceMode.ABDUCTION: "Generating hypotheses to explain surprising facts",
            InferenceMode.QUALITATIVE: "Reasoning from felt qualities and intuition",
            InferenceMode.EQUILIBRIUM: "Finding where dynamic processes converge",
            InferenceMode.DEDUCTION: "Deriving necessary conclusions from premises",
            InferenceMode.SYNTACTIC: "Analyzing organizational structure",
            InferenceMode.DIAGRAMMATIC: "Reasoning through diagram manipulation",
            InferenceMode.METONYMIC: "Following conventional associations",
            InferenceMode.TYPE_TOKEN: "Transferring properties from type to instance",
            InferenceMode.INDUCTION: "Generalizing patterns from instances",
            InferenceMode.ANALOGICAL: "Mapping structure between domains",
            InferenceMode.INDEXICAL: "Tracking from effects to causes",
            InferenceMode.CONTRASTIVE: "Learning from differences between concepts",
        }
        
        candidates_str = "\n".join([
            f"  {i+1}. {mode.name}: {mode_descriptions[mode]}"
            for i, (mode, _) in enumerate(top_candidates)
        ])
        
        prompt = f"""Classify this reasoning task into the most appropriate inference mode.

Task: {task}
Context: {context}

Top candidate modes:
{candidates_str}

Which mode best fits this task? Respond with:
MODE: [mode name]
CONFIDENCE: [0.0-1.0]
REASONING: [why this mode]"""

        response = self.llm_call(prompt)
        
        # Parse response
        mode_name = "ABDUCTION"  # default
        confidence = 0.5
        reasoning = ""
        
        for line in response.split("\n"):
            if line.startswith("MODE:"):
                mode_name = line.replace("MODE:", "").strip().upper()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        # Find the mode
        try:
            mode = InferenceMode[mode_name]
        except KeyError:
            mode = top_candidates[0][0]  # Fall back to top pattern match
        
        return ClassificationResult(
            mode=mode,
            architecture=MODE_TO_ARCHITECTURE[mode],
            confidence=confidence,
            signals=[f"LLM: {reasoning}"],
            alternative_modes=[(m, s) for m, (s, _) in top_candidates[1:3]]
        )


class ReasoningRouter:
    """
    Routes reasoning tasks to the appropriate architecture and mode.
    
    This is the main entry point for the multi-modal inference system.
    """
    
    def __init__(
        self, 
        llm_call: Callable[[str], str],
        randomness_source: Optional[Callable[[], float]] = None
    ):
        """
        Initialize the router.
        
        Args:
            llm_call: Function to call the LLM
            randomness_source: Optional custom randomness source (e.g., QRNG)
        """
        self.llm_call = llm_call
        self.randomness_source = randomness_source
        
        # Initialize classifier
        self.classifier = ModeClassifier(llm_call=llm_call)
        
        # Initialize architectures
        self.architectures: Dict[Architecture, BaseArchitecture] = {
            Architecture.STRANGE_ATTRACTOR: StrangeAttractorArchitecture(randomness_source),
            Architecture.CODE_DUALITY: CodeDualityArchitecture(randomness_source),
            Architecture.TENSEGRITY: TensegrityArchitecture(randomness_source),
        }
    
    def reason(
        self,
        task: str,
        inputs: Optional[Dict[str, Any]] = None,
        force_mode: Optional[InferenceMode] = None,
        max_iterations: int = 10,
        use_llm_classification: bool = False
    ) -> Dict[str, Any]:
        """
        Route a reasoning task to the appropriate mode and execute inference.
        
        Args:
            task: The reasoning task or question
            inputs: Additional inputs for the inference
            force_mode: Override automatic classification
            max_iterations: Maximum iterations for iterative modes
            use_llm_classification: Use LLM for ambiguous classification
            
        Returns:
            Dictionary with classification, result, and metadata
        """
        inputs = inputs or {}
        
        # Classify the task
        if force_mode:
            classification = ClassificationResult(
                mode=force_mode,
                architecture=MODE_TO_ARCHITECTURE[force_mode],
                confidence=1.0,
                signals=["forced"],
                alternative_modes=[]
            )
        else:
            classification = self.classifier.classify(
                task, 
                context=inputs,
                use_llm=use_llm_classification
            )
        
        # Get the appropriate architecture
        architecture = self.architectures[classification.architecture]
        
        # Prepare inputs based on mode
        mode_inputs = self._prepare_inputs(task, inputs, classification.mode)
        
        # Execute inference
        result = architecture.infer(
            mode=classification.mode,
            inputs=mode_inputs,
            llm_call=self.llm_call,
            max_iterations=max_iterations
        )
        
        return {
            "task": task,
            "classification": {
                "mode": classification.mode.name,
                "architecture": classification.architecture.name,
                "confidence": classification.confidence,
                "signals": classification.signals,
                "alternatives": [
                    {"mode": m.name, "score": s} 
                    for m, s in classification.alternative_modes
                ]
            },
            "result": result.to_dict(),
            "metadata": {
                "random_calls": architecture.random_call_count,
                "iterations": result.state.iteration,
                "converged": result.state.converged
            }
        }
    
    def _prepare_inputs(
        self, 
        task: str, 
        inputs: Dict[str, Any], 
        mode: InferenceMode
    ) -> Dict[str, Any]:
        """Prepare mode-specific inputs."""
        # Merge task into inputs if not already present
        prepared = inputs.copy()
        
        # Add task as default for certain fields based on mode
        if mode == InferenceMode.ABDUCTION and "observation" not in prepared:
            prepared["observation"] = task
        elif mode == InferenceMode.QUALITATIVE and "situation" not in prepared:
            prepared["situation"] = task
        elif mode == InferenceMode.EQUILIBRIUM and "system" not in prepared:
            prepared["system"] = task
        elif mode == InferenceMode.DEDUCTION and "premises" not in prepared:
            prepared["premises"] = [task]
        elif mode == InferenceMode.SYNTACTIC and "structure" not in prepared:
            prepared["structure"] = task
        elif mode == InferenceMode.INDUCTION and "domain" not in prepared:
            prepared["domain"] = task
        elif mode == InferenceMode.CONTRASTIVE:
            if "concept_a" not in prepared and "concept_b" not in prepared:
                # Try to extract from task
                prepared["concept_a"] = task
                prepared["concept_b"] = ""
        
        return prepared
    
    def get_architecture_stats(self) -> Dict[str, int]:
        """Get random call counts for each architecture."""
        return {
            arch.name: self.architectures[arch].random_call_count
            for arch in Architecture
        }
    
    def reset_stats(self):
        """Reset random call counters for all architectures."""
        for arch in self.architectures.values():
            arch.reset_random_counter()
