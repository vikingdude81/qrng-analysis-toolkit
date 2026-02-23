#!/usr/bin/env python3
"""
QRNG vs PRNG Inference Pilot Experiment
========================================
Tests whether quantum randomness affects LLM inference dynamics
differently than pseudo-randomness.

Hypothesis: Strange Attractor architecture (which uses randomness for
convergence/exploration) may show different behavior with QRNG vs PRNG.

Metrics:
- Convergence iterations
- Final confidence
- Response consistency (semantic similarity)
- Token usage patterns
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
import statistics

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import numpy as np

from inference_framework import (
    QRNGSourceType,
    UnifiedRandomnessProvider,
)

console = Console()


@dataclass
class InferenceResult:
    """Result from a single inference trial."""
    source_type: str
    trial_id: int
    prompt: str
    response: str
    response_hash: str
    iterations: int
    final_confidence: float
    convergence_time_ms: float
    tokens_used: int
    rng_calls: int
    rng_values_used: list = field(default_factory=list)


@dataclass 
class ExperimentConfig:
    """Configuration for the experiment."""
    trials_per_condition: int = 5
    max_iterations: int = 10
    convergence_threshold: float = 0.85
    temperature: float = 0.7
    model: str = "claude-sonnet-4-20250514"


class StrangeAttractorSimulator:
    """
    Simulates Strange Attractor inference dynamics.
    
    Uses randomness for:
    1. Initial condition perturbation
    2. Exploration during convergence
    3. Confidence jitter (simulating attractor basin dynamics)
    """
    
    def __init__(
        self, 
        llm_call: Callable[[str], tuple[str, int]],
        rng_source: Callable[[], float],
        config: ExperimentConfig
    ):
        self.llm_call = llm_call
        self.rng = rng_source
        self.config = config
        self.rng_calls = 0
        self.rng_values = []
    
    def _sample_rng(self) -> float:
        """Sample from RNG and track usage."""
        val = self.rng()
        self.rng_calls += 1
        self.rng_values.append(val)
        return val
    
    def _perturb_prompt(self, prompt: str) -> str:
        """Add subtle perturbation based on RNG (simulates initial conditions)."""
        # Use RNG to select perturbation style
        style_val = self._sample_rng()
        
        styles = [
            "",  # No perturbation
            " Think step by step.",
            " Consider multiple perspectives.",
            " Be precise and analytical.",
            " Think creatively.",
        ]
        
        style_idx = int(style_val * len(styles))
        style_idx = min(style_idx, len(styles) - 1)
        
        return prompt + styles[style_idx]
    
    def _compute_confidence(self, response: str, iteration: int) -> float:
        """
        Compute confidence score with RNG-based jitter.
        Simulates attractor basin dynamics.
        """
        # Base confidence increases with response quality indicators
        base = 0.5
        
        # Length bonus (more thorough = higher confidence)
        length_bonus = min(len(response) / 1000, 0.2)
        
        # Structure bonus (paragraphs, lists)
        structure_bonus = 0.1 if '\n\n' in response else 0.0
        
        # Iteration decay (converging)
        iteration_factor = 1.0 - (0.5 ** iteration)
        
        # RNG jitter (attractor basin noise)
        jitter = (self._sample_rng() - 0.5) * 0.1
        
        confidence = base + length_bonus + structure_bonus
        confidence *= iteration_factor
        confidence += jitter
        
        return max(0.0, min(1.0, confidence))
    
    def _should_continue(self, confidence: float) -> bool:
        """
        Decide whether to continue iterating.
        Uses RNG for exploration/exploitation tradeoff.
        """
        if confidence >= self.config.convergence_threshold:
            # High confidence - but RNG might push for more exploration
            explore_val = self._sample_rng()
            return explore_val > 0.8  # 20% chance to continue exploring
        else:
            # Low confidence - but RNG might accept early
            accept_val = self._sample_rng()
            return accept_val > 0.3  # 70% chance to continue
    
    def run_inference(self, prompt: str) -> tuple[str, int, float, int]:
        """
        Run Strange Attractor inference loop.
        
        Returns: (final_response, iterations, final_confidence, tokens_used)
        """
        self.rng_calls = 0
        self.rng_values = []
        
        # Perturb initial prompt (sensitive dependence on initial conditions)
        perturbed_prompt = self._perturb_prompt(prompt)
        
        best_response = ""
        best_confidence = 0.0
        total_tokens = 0
        
        for iteration in range(1, self.config.max_iterations + 1):
            # Make LLM call
            response, tokens = self.llm_call(perturbed_prompt)
            total_tokens += tokens
            
            # Compute confidence with RNG jitter
            confidence = self._compute_confidence(response, iteration)
            
            if confidence > best_confidence:
                best_response = response
                best_confidence = confidence
            
            # Check convergence with RNG-influenced decision
            if not self._should_continue(confidence):
                break
            
            # Modify prompt for next iteration (attractor dynamics)
            refinement_val = self._sample_rng()
            if refinement_val > 0.5:
                perturbed_prompt = f"Refine this response:\n{response[:500]}\n\nOriginal question: {prompt}"
        
        return best_response, iteration, best_confidence, total_tokens


def create_claude_caller(config: ExperimentConfig) -> Callable[[str], tuple[str, int]]:
    """Create a Claude API caller function."""
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from env
    
    def call_claude(prompt: str) -> tuple[str, int]:
        """Call Claude and return (response, tokens_used)."""
        message = client.messages.create(
            model=config.model,
            max_tokens=1024,
            temperature=config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = message.content[0].text
        tokens = message.usage.input_tokens + message.usage.output_tokens
        return response_text, tokens
    
    return call_claude


def run_trial(
    source_type: QRNGSourceType,
    trial_id: int,
    prompt: str,
    provider: UnifiedRandomnessProvider,
    config: ExperimentConfig,
    llm_call: Callable[[str], tuple[str, int]]
) -> InferenceResult:
    """Run a single trial with the specified randomness source."""
    
    # Get RNG source (use trial_id as seed for reproducibility with PRNG)
    rng_source = provider.get_source(source_type, seed=trial_id * 12345)
    
    # Create simulator
    simulator = StrangeAttractorSimulator(llm_call, rng_source, config)
    
    # Run inference
    start_time = time.perf_counter()
    response, iterations, confidence, tokens = simulator.run_inference(prompt)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Compute response hash for consistency comparison
    response_hash = hashlib.sha256(response.encode()).hexdigest()[:16]
    
    return InferenceResult(
        source_type=source_type.name,
        trial_id=trial_id,
        prompt=prompt,
        response=response,
        response_hash=response_hash,
        iterations=iterations,
        final_confidence=confidence,
        convergence_time_ms=elapsed_ms,
        tokens_used=tokens,
        rng_calls=simulator.rng_calls,
        rng_values_used=simulator.rng_values[:10]  # Store first 10 for analysis
    )


def analyze_results(results: list[InferenceResult]) -> dict:
    """Analyze results by condition."""
    
    # Group by source type
    by_source = {}
    for r in results:
        if r.source_type not in by_source:
            by_source[r.source_type] = []
        by_source[r.source_type].append(r)
    
    analysis = {}
    for source, trials in by_source.items():
        iterations = [t.iterations for t in trials]
        confidences = [t.final_confidence for t in trials]
        times = [t.convergence_time_ms for t in trials]
        tokens = [t.tokens_used for t in trials]
        rng_calls = [t.rng_calls for t in trials]
        
        # Response consistency (unique hashes)
        unique_hashes = len(set(t.response_hash for t in trials))
        
        analysis[source] = {
            "n": len(trials),
            "iterations_mean": statistics.mean(iterations),
            "iterations_std": statistics.stdev(iterations) if len(iterations) > 1 else 0,
            "confidence_mean": statistics.mean(confidences),
            "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "time_ms_mean": statistics.mean(times),
            "tokens_mean": statistics.mean(tokens),
            "rng_calls_mean": statistics.mean(rng_calls),
            "unique_responses": unique_hashes,
            "consistency_ratio": 1.0 - (unique_hashes - 1) / max(len(trials) - 1, 1)
        }
    
    # Compute effect sizes (Cohen's d) between QRNG and PRNG
    if "OUTSHIFT_STREAM" in analysis and "PRNG" in analysis:
        qrng = by_source["OUTSHIFT_STREAM"]
        prng = by_source["PRNG"]
        
        qrng_iters = [t.iterations for t in qrng]
        prng_iters = [t.iterations for t in prng]
        
        if len(qrng_iters) > 1 and len(prng_iters) > 1:
            pooled_std = np.sqrt(
                (statistics.stdev(qrng_iters)**2 + statistics.stdev(prng_iters)**2) / 2
            )
            if pooled_std > 0:
                cohens_d = (statistics.mean(qrng_iters) - statistics.mean(prng_iters)) / pooled_std
                analysis["effect_size_iterations"] = cohens_d
    
    return analysis


def print_results(results: list[InferenceResult], analysis: dict):
    """Print formatted results."""
    
    console.print(Panel.fit("📊 EXPERIMENT RESULTS", style="bold green"))
    
    # Summary table
    table = Table(title="Results by Randomness Source")
    table.add_column("Source", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Iterations", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Time (ms)", justify="right")
    table.add_column("RNG Calls", justify="right")
    table.add_column("Consistency", justify="right")
    
    for source, stats in analysis.items():
        if source.startswith("effect_"):
            continue
        table.add_row(
            source,
            str(stats["n"]),
            f"{stats['iterations_mean']:.1f} ± {stats['iterations_std']:.1f}",
            f"{stats['confidence_mean']:.3f}",
            f"{stats['time_ms_mean']:.0f}",
            f"{stats['rng_calls_mean']:.1f}",
            f"{stats['consistency_ratio']:.2f}"
        )
    
    console.print(table)
    
    # Effect size
    if "effect_size_iterations" in analysis:
        d = analysis["effect_size_iterations"]
        magnitude = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        console.print(f"\n[bold]Effect Size (QRNG vs PRNG iterations):[/] Cohen's d = {d:.3f} ({magnitude})")


def save_results(results: list[InferenceResult], analysis: dict, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"pilot_experiment_{timestamp}.json"
    
    # Convert results to serializable format
    results_data = []
    for r in results:
        d = asdict(r)
        # Truncate response for storage
        d["response"] = d["response"][:500] + "..." if len(d["response"]) > 500 else d["response"]
        results_data.append(d)
    
    data = {
        "timestamp": timestamp,
        "experiment": "qrng_vs_prng_strange_attractor",
        "results": results_data,
        "analysis": analysis
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    console.print(f"\n[green]Results saved to:[/] {output_file}")
    return output_file


# Test prompts that benefit from exploration/convergence
TEST_PROMPTS = [
    "What is the relationship between entropy and information?",
    "Explain why the halting problem is undecidable.",
    "What are the philosophical implications of quantum superposition?",
]


def main():
    console.print(Panel.fit(
        "🔬 QRNG vs PRNG INFERENCE PILOT EXPERIMENT",
        style="bold magenta"
    ))
    
    # Configuration
    config = ExperimentConfig(
        trials_per_condition=10,  # 5 sources x 10 = 50 trials
        max_iterations=5,
        convergence_threshold=0.8,
        temperature=0.7,
        model="claude-sonnet-4-20250514"
    )
    
    console.print(f"\n[bold]Configuration:[/]")
    console.print(f"  Trials per condition: {config.trials_per_condition}")
    console.print(f"  Max iterations: {config.max_iterations}")
    console.print(f"  Model: {config.model}")
    console.print(f"  Temperature: {config.temperature}")
    
    # Initialize provider
    provider = UnifiedRandomnessProvider()
    console.print(f"\n[bold]QRNG Pool:[/] {provider.stream_stats.count} samples available")
    
    # Create LLM caller
    console.print("\n[bold]Initializing Claude API...[/]")
    try:
        llm_call = create_claude_caller(config)
    except Exception as e:
        console.print(f"[red]Error initializing Claude:[/] {e}")
        return
    
    # Conditions to test
    conditions = [
        QRNGSourceType.OUTSHIFT_STREAM,
        QRNGSourceType.ANU_QRNG,
        QRNGSourceType.CIPHERSTONE_QRNG,
        QRNGSourceType.CPU_RDRAND,
        QRNGSourceType.PRNG,
    ]
    
    # Run experiment
    all_results = []
    prompt = TEST_PROMPTS[0]  # Use first prompt for pilot
    
    console.print(f"\n[bold]Test Prompt:[/] {prompt[:60]}...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for source_type in conditions:
            task = progress.add_task(
                f"Running {source_type.name}...", 
                total=config.trials_per_condition
            )
            
            for trial_id in range(config.trials_per_condition):
                progress.update(task, description=f"{source_type.name} trial {trial_id + 1}")
                
                try:
                    result = run_trial(
                        source_type=source_type,
                        trial_id=trial_id,
                        prompt=prompt,
                        provider=provider,
                        config=config,
                        llm_call=llm_call
                    )
                    all_results.append(result)
                    console.print(f"  ✓ Trial {trial_id + 1}: {result.iterations} iters, conf={result.final_confidence:.3f}")
                except Exception as e:
                    console.print(f"  [red]✗ Trial {trial_id + 1} failed:[/] {e}")
                
                progress.advance(task)
    
    if not all_results:
        console.print("[red]No results collected![/]")
        return
    
    # Analyze
    analysis = analyze_results(all_results)
    
    # Print results
    print_results(all_results, analysis)
    
    # Save
    output_dir = Path(__file__).parent / "inference_results"
    save_results(all_results, analysis, output_dir)
    
    console.print("\n[bold green]✓ Pilot experiment complete![/]")


if __name__ == "__main__":
    main()
