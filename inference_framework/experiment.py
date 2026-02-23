"""
QRNG Experiment Module: Testing Entropy Sensitivity Across Inference Architectures

This module implements the experimental framework for investigating whether
different inference architectures show differential sensitivity to the source
of randomness (QRNG vs PRNG).

Hypothesis: Strange Attractor and Tensegrity architectures will show measurable
differences in convergence dynamics when using QRNG vs PRNG, while Code Duality
(being more deterministic) will show no significant difference.
"""

import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import statistics
import random

from .architectures import Architecture, InferenceMode, MODE_TO_ARCHITECTURE
from .classifier import ReasoningRouter


class RandomnessSource(Enum):
    """Types of randomness sources."""
    QRNG = "qrng"                   # Quantum random number generator
    PRNG_SECURE = "prng_secure"    # Cryptographically secure PRNG
    PRNG_FAST = "prng_fast"        # Fast PRNG (Mersenne Twister)


@dataclass
class TrialResult:
    """Result of a single experimental trial."""
    trial_id: str
    task_id: str
    mode: str
    architecture: str
    randomness_source: str
    
    # Primary measures
    conclusion: str
    confidence: float
    iterations: int
    converged: bool
    random_calls: int
    
    # Timing
    duration_ms: float
    
    # Convergence path (for Strange Attractor and Tensegrity)
    intermediate_confidences: List[float] = field(default_factory=list)
    
    # Hash of conclusion for comparing across runs
    conclusion_hash: str = ""
    
    def __post_init__(self):
        self.conclusion_hash = hashlib.md5(self.conclusion.encode()).hexdigest()[:8]


@dataclass
class TaskDefinition:
    """Definition of an experimental task."""
    task_id: str
    mode: InferenceMode
    task_text: str
    inputs: Dict[str, Any]
    expected_architecture: Architecture
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "mode": self.mode.name,
            "task_text": self.task_text,
            "expected_architecture": self.expected_architecture.name
        }


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    experiment_id: str
    trials_per_condition: int = 10
    randomness_sources: List[RandomnessSource] = field(
        default_factory=lambda: [
            RandomnessSource.QRNG,
            RandomnessSource.PRNG_SECURE,
            RandomnessSource.PRNG_FAST
        ]
    )
    max_iterations: int = 10
    seed_base: int = 42  # For reproducible PRNG conditions


class RandomnessProvider:
    """
    Provides different sources of randomness for experiments.
    
    In practice, QRNG would connect to actual quantum hardware.
    This implementation includes a mock QRNG that can be replaced.
    """
    
    def __init__(self, qrng_source: Optional[Callable[[], float]] = None):
        """
        Initialize with optional custom QRNG source.
        
        Args:
            qrng_source: Function returning quantum random numbers.
                        If None, uses simulated "QRNG" for testing.
        """
        self._qrng_source = qrng_source
        self._prng_secure = random.SystemRandom()
        self._prng_fast = random.Random()
        
    def get_source(
        self, 
        source_type: RandomnessSource,
        seed: Optional[int] = None
    ) -> Callable[[], float]:
        """
        Get a randomness source function.
        
        Args:
            source_type: Type of randomness to use
            seed: Seed for PRNG sources (ignored for QRNG)
            
        Returns:
            Function that returns random floats in [0, 1)
        """
        if source_type == RandomnessSource.QRNG:
            if self._qrng_source:
                return self._qrng_source
            else:
                # Mock QRNG - in practice, replace with real hardware
                return self._mock_qrng
        
        elif source_type == RandomnessSource.PRNG_SECURE:
            # SystemRandom doesn't support seeding
            return self._prng_secure.random
        
        elif source_type == RandomnessSource.PRNG_FAST:
            if seed is not None:
                self._prng_fast.seed(seed)
            return self._prng_fast.random
        
        raise ValueError(f"Unknown source type: {source_type}")
    
    def _mock_qrng(self) -> float:
        """
        Mock QRNG for testing.
        
        In production, this would be replaced with actual quantum hardware,
        such as:
        - ANU QRNG API
        - ID Quantique hardware
        - IBM Quantum random
        """
        # For testing: use SystemRandom as a stand-in
        # Mark with slight timing delay to simulate hardware call
        time.sleep(0.001)  # 1ms simulated hardware delay
        return self._prng_secure.random()


# Standard task battery for experiments
STANDARD_TASKS: List[TaskDefinition] = [
    # Strange Attractor tasks
    TaskDefinition(
        task_id="SA_ABD_01",
        mode=InferenceMode.ABDUCTION,
        task_text="The patient presents with fatigue, weight gain, and cold intolerance. What would explain these symptoms?",
        inputs={"context": "Medical diagnosis scenario"},
        expected_architecture=Architecture.STRANGE_ATTRACTOR
    ),
    TaskDefinition(
        task_id="SA_ABD_02",
        mode=InferenceMode.ABDUCTION,
        task_text="A company's stock price dropped 15% despite positive earnings. What would explain this?",
        inputs={"context": "Financial analysis"},
        expected_architecture=Architecture.STRANGE_ATTRACTOR
    ),
    TaskDefinition(
        task_id="SA_QUAL_01",
        mode=InferenceMode.QUALITATIVE,
        task_text="What do the qualities of this startup's pitch intimate about its likely success?",
        inputs={"qualities": ["ambitious", "vague", "passionate", "underfunded"]},
        expected_architecture=Architecture.STRANGE_ATTRACTOR
    ),
    TaskDefinition(
        task_id="SA_EQ_01",
        mode=InferenceMode.EQUILIBRIUM,
        task_text="Where does the social media platform's engagement dynamics converge?",
        inputs={
            "dynamics": "Users seek engagement, platform optimizes for time-on-site, content becomes more sensational",
            "initial_conditions": "New platform with quality focus"
        },
        expected_architecture=Architecture.STRANGE_ATTRACTOR
    ),
    
    # Code Duality tasks
    TaskDefinition(
        task_id="CD_DED_01",
        mode=InferenceMode.DEDUCTION,
        task_text="Given: All mammals are warm-blooded. All whales are mammals. What must follow?",
        inputs={"premises": ["All mammals are warm-blooded", "All whales are mammals"]},
        expected_architecture=Architecture.CODE_DUALITY
    ),
    TaskDefinition(
        task_id="CD_DED_02",
        mode=InferenceMode.DEDUCTION,
        task_text="If the API returns 200, the request succeeded. The API returned 200. What must follow?",
        inputs={"premises": ["If API returns 200, request succeeded", "API returned 200"]},
        expected_architecture=Architecture.CODE_DUALITY
    ),
    TaskDefinition(
        task_id="CD_SYN_01",
        mode=InferenceMode.SYNTACTIC,
        task_text="How is this sentence organized: 'The quick brown fox jumps over the lazy dog'?",
        inputs={"structure": "The quick brown fox jumps over the lazy dog"},
        expected_architecture=Architecture.CODE_DUALITY
    ),
    TaskDefinition(
        task_id="CD_TT_01",
        mode=InferenceMode.TYPE_TOKEN,
        task_text="What properties transfer from the type 'RESTful API' to this specific instance?",
        inputs={
            "type": "RESTful API: stateless, resource-based, uses HTTP methods",
            "token": "Twitter's v2 API"
        },
        expected_architecture=Architecture.CODE_DUALITY
    ),
    
    # Tensegrity tasks
    TaskDefinition(
        task_id="TE_IND_01",
        mode=InferenceMode.INDUCTION,
        task_text="What pattern holds across these successful startups?",
        inputs={
            "instances": [
                "Stripe: solved payments complexity with simple API",
                "Slack: replaced email for team communication",
                "Figma: made design collaborative in browser"
            ],
            "domain": "B2B software startups"
        },
        expected_architecture=Architecture.TENSEGRITY
    ),
    TaskDefinition(
        task_id="TE_ANA_01",
        mode=InferenceMode.ANALOGICAL,
        task_text="What corresponds between the solar system and the atom?",
        inputs={
            "source": {"domain": "solar system", "structure": "sun at center, planets orbit"},
            "target": {"domain": "atom", "structure": "nucleus at center, electrons...?"}
        },
        expected_architecture=Architecture.TENSEGRITY
    ),
    TaskDefinition(
        task_id="TE_IDX_01",
        mode=InferenceMode.INDEXICAL,
        task_text="The server logs show 500 errors spiking at 3am. What does this trace indicate?",
        inputs={
            "sign": "500 errors spiking at 3am daily",
            "context": "Production web server, no scheduled jobs at that time"
        },
        expected_architecture=Architecture.TENSEGRITY
    ),
    TaskDefinition(
        task_id="TE_CON_01",
        mode=InferenceMode.CONTRASTIVE,
        task_text="What does the difference between machine learning and traditional programming reveal?",
        inputs={
            "concept_a": "Traditional programming: explicit rules written by humans",
            "concept_b": "Machine learning: patterns learned from data"
        },
        expected_architecture=Architecture.TENSEGRITY
    ),
]


class InferenceExperiment:
    """
    Runs controlled experiments comparing inference across randomness sources.
    """
    
    def __init__(
        self,
        llm_call: Callable[[str], str],
        config: ExperimentConfig,
        qrng_source: Optional[Callable[[], float]] = None
    ):
        """
        Initialize the experiment.
        
        Args:
            llm_call: Function to call the LLM
            config: Experiment configuration
            qrng_source: Optional actual QRNG hardware interface
        """
        self.llm_call = llm_call
        self.config = config
        self.randomness_provider = RandomnessProvider(qrng_source)
        self.results: List[TrialResult] = []
        
    def run_experiment(
        self, 
        tasks: Optional[List[TaskDefinition]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run the full experiment across all conditions.
        
        Args:
            tasks: Tasks to use (defaults to STANDARD_TASKS)
            verbose: Print progress
            
        Returns:
            Experiment results with analysis
        """
        tasks = tasks or STANDARD_TASKS
        
        if verbose:
            print(f"Starting experiment {self.config.experiment_id}")
            print(f"Tasks: {len(tasks)}")
            print(f"Trials per condition: {self.config.trials_per_condition}")
            print(f"Randomness sources: {[s.name for s in self.config.randomness_sources]}")
        
        # Run trials
        for task in tasks:
            for source in self.config.randomness_sources:
                for trial_num in range(self.config.trials_per_condition):
                    trial_id = f"{task.task_id}_{source.name}_{trial_num:03d}"
                    
                    # Seed for reproducibility (PRNG only)
                    seed = self.config.seed_base + hash(trial_id) % (2**31)
                    
                    result = self._run_trial(task, source, trial_id, seed)
                    self.results.append(result)
                    
                    if verbose:
                        print(f"  Completed: {trial_id} -> {result.converged}, {result.iterations} iter")
        
        # Analyze results
        analysis = self._analyze_results()
        
        return {
            "experiment_id": self.config.experiment_id,
            "config": {
                "experiment_id": self.config.experiment_id,
                "trials_per_condition": self.config.trials_per_condition,
                "max_iterations": self.config.max_iterations,
                "seed_base": self.config.seed_base,
                "randomness_sources": [s.value for s in self.config.randomness_sources]
            },
            "tasks": [t.to_dict() for t in tasks],
            "results": [asdict(r) for r in self.results],
            "analysis": analysis
        }
    
    def _run_trial(
        self,
        task: TaskDefinition,
        source: RandomnessSource,
        trial_id: str,
        seed: int
    ) -> TrialResult:
        """Run a single trial."""
        
        # Get randomness source
        random_fn = self.randomness_provider.get_source(source, seed)
        
        # Create router with this randomness source
        router = ReasoningRouter(
            llm_call=self.llm_call,
            randomness_source=random_fn
        )
        
        # Run inference
        start_time = time.time()
        
        result = router.reason(
            task=task.task_text,
            inputs=task.inputs,
            force_mode=task.mode,
            max_iterations=self.config.max_iterations
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Extract intermediate confidences if available
        intermediate_confidences = []
        state = result.get("result", {}).get("state", {})
        if "intermediate_results" in state:
            for ir in state["intermediate_results"]:
                if isinstance(ir, dict) and "confidence" in ir:
                    intermediate_confidences.append(ir["confidence"])
        
        return TrialResult(
            trial_id=trial_id,
            task_id=task.task_id,
            mode=task.mode.name,
            architecture=task.expected_architecture.name,
            randomness_source=source.value,
            conclusion=result["result"]["conclusion"],
            confidence=result["result"]["state"]["confidence"],
            iterations=result["metadata"]["iterations"],
            converged=result["metadata"]["converged"],
            random_calls=result["metadata"]["random_calls"],
            duration_ms=duration_ms,
            intermediate_confidences=intermediate_confidences
        )
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        
        analysis = {
            "by_architecture": {},
            "by_source": {},
            "interaction_effects": {},
            "hypothesis_test": {}
        }
        
        # Group results
        by_arch: Dict[str, List[TrialResult]] = {}
        by_source: Dict[str, List[TrialResult]] = {}
        by_arch_source: Dict[str, List[TrialResult]] = {}
        
        for r in self.results:
            # By architecture
            if r.architecture not in by_arch:
                by_arch[r.architecture] = []
            by_arch[r.architecture].append(r)
            
            # By source
            if r.randomness_source not in by_source:
                by_source[r.randomness_source] = []
            by_source[r.randomness_source].append(r)
            
            # By interaction
            key = f"{r.architecture}_{r.randomness_source}"
            if key not in by_arch_source:
                by_arch_source[key] = []
            by_arch_source[key].append(r)
        
        # Analyze by architecture
        for arch, results in by_arch.items():
            analysis["by_architecture"][arch] = self._compute_stats(results)
        
        # Analyze by source
        for source, results in by_source.items():
            analysis["by_source"][source] = self._compute_stats(results)
        
        # Analyze interactions
        for key, results in by_arch_source.items():
            analysis["interaction_effects"][key] = self._compute_stats(results)
        
        # Test the hypothesis
        analysis["hypothesis_test"] = self._test_hypothesis(by_arch_source)
        
        return analysis
    
    def _compute_stats(self, results: List[TrialResult]) -> Dict[str, Any]:
        """Compute summary statistics for a group of results."""
        if not results:
            return {}
        
        iterations = [r.iterations for r in results]
        confidences = [r.confidence for r in results]
        random_calls = [r.random_calls for r in results]
        durations = [r.duration_ms for r in results]
        convergence_rate = sum(1 for r in results if r.converged) / len(results)
        
        # Solution diversity (unique conclusions)
        unique_conclusions = len(set(r.conclusion_hash for r in results))
        
        return {
            "n": len(results),
            "iterations_mean": statistics.mean(iterations),
            "iterations_std": statistics.stdev(iterations) if len(iterations) > 1 else 0,
            "confidence_mean": statistics.mean(confidences),
            "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "random_calls_mean": statistics.mean(random_calls),
            "duration_ms_mean": statistics.mean(durations),
            "convergence_rate": convergence_rate,
            "solution_diversity": unique_conclusions / len(results)
        }
    
    def _test_hypothesis(
        self, 
        by_arch_source: Dict[str, List[TrialResult]]
    ) -> Dict[str, Any]:
        """
        Test the main hypothesis: Strange Attractor and Tensegrity show
        differential sensitivity to randomness source, Code Duality does not.
        """
        
        def get_metrics_by_source(arch: str, metric: str) -> Dict[str, List[float]]:
            """Get metric values grouped by randomness source for an architecture."""
            result = {}
            for source in ["qrng", "prng_secure", "prng_fast"]:
                key = f"{arch}_{source}"
                if key in by_arch_source:
                    result[source] = [
                        getattr(r, metric) for r in by_arch_source[key]
                    ]
            return result
        
        def compute_effect_size(group1: List[float], group2: List[float]) -> float:
            """Compute Cohen's d effect size."""
            if not group1 or not group2 or len(group1) < 2 or len(group2) < 2:
                return 0.0
            n1, n2 = len(group1), len(group2)
            var1 = statistics.variance(group1)
            var2 = statistics.variance(group2)
            pooled_std = ((var1 * (n1-1) + var2 * (n2-1)) / (n1 + n2 - 2)) ** 0.5
            if pooled_std == 0:
                return 0.0
            return (statistics.mean(group1) - statistics.mean(group2)) / pooled_std
        
        hypothesis_results = {}
        
        for arch in ["STRANGE_ATTRACTOR", "CODE_DUALITY", "TENSEGRITY"]:
            arch_results = {
                "iterations": {},
                "confidence": {},
                "solution_diversity": {},
                "effect_sizes": {}
            }
            
            # Get metrics by source
            iterations_by_source = get_metrics_by_source(arch, "iterations")
            confidence_by_source = get_metrics_by_source(arch, "confidence")
            
            # Compute means
            for metric_name, by_source in [
                ("iterations", iterations_by_source),
                ("confidence", confidence_by_source)
            ]:
                for source, values in by_source.items():
                    arch_results[metric_name][source] = {
                        "mean": statistics.mean(values) if values else 0,
                        "std": statistics.stdev(values) if len(values) > 1 else 0
                    }
            
            # Effect sizes: QRNG vs PRNG
            if "qrng" in iterations_by_source and "prng_fast" in iterations_by_source:
                arch_results["effect_sizes"]["iterations_qrng_vs_prng"] = compute_effect_size(
                    iterations_by_source["qrng"],
                    iterations_by_source["prng_fast"]
                )
            
            if "qrng" in confidence_by_source and "prng_fast" in confidence_by_source:
                arch_results["effect_sizes"]["confidence_qrng_vs_prng"] = compute_effect_size(
                    confidence_by_source["qrng"],
                    confidence_by_source["prng_fast"]
                )
            
            # Solution diversity by source
            for source in ["qrng", "prng_secure", "prng_fast"]:
                key = f"{arch}_{source}"
                if key in by_arch_source:
                    results = by_arch_source[key]
                    unique = len(set(r.conclusion_hash for r in results))
                    arch_results["solution_diversity"][source] = unique / len(results) if results else 0
            
            hypothesis_results[arch] = arch_results
        
        # Overall hypothesis assessment
        sa_effect = hypothesis_results.get("STRANGE_ATTRACTOR", {}).get("effect_sizes", {})
        cd_effect = hypothesis_results.get("CODE_DUALITY", {}).get("effect_sizes", {})
        te_effect = hypothesis_results.get("TENSEGRITY", {}).get("effect_sizes", {})
        
        sa_iter_effect = abs(sa_effect.get("iterations_qrng_vs_prng", 0))
        cd_iter_effect = abs(cd_effect.get("iterations_qrng_vs_prng", 0))
        te_iter_effect = abs(te_effect.get("iterations_qrng_vs_prng", 0))
        
        hypothesis_supported = (
            sa_iter_effect > cd_iter_effect and
            te_iter_effect > cd_iter_effect
        )
        
        return {
            "by_architecture": hypothesis_results,
            "summary": {
                "strange_attractor_effect": sa_iter_effect,
                "code_duality_effect": cd_iter_effect,
                "tensegrity_effect": te_iter_effect
            },
            "hypothesis_supported": hypothesis_supported,
            "interpretation": (
                "Strange Attractor and Tensegrity architectures show larger effect sizes "
                "for QRNG vs PRNG differences compared to Code Duality, "
                "supporting the hypothesis of differential entropy sensitivity."
                if hypothesis_supported else
                "Effect sizes do not clearly support differential entropy sensitivity hypothesis. "
                "This may indicate: (1) more trials needed, (2) different tasks required, or "
                "(3) the hypothesis needs refinement."
            )
        }
    
    def save_results(self, filepath: str):
        """Save experiment results to JSON."""
        results = {
            "experiment_id": self.config.experiment_id,
            "results": [asdict(r) for r in self.results],
            "analysis": self._analyze_results()
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    @classmethod
    def load_results(cls, filepath: str) -> Dict[str, Any]:
        """Load experiment results from JSON."""
        with open(filepath, 'r') as f:
            return json.load(f)
