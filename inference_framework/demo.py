#!/usr/bin/env python3
"""
Demo: Multi-Modal Inference System

This script demonstrates the reasoning router and mode classifier in action.
It shows how different types of reasoning tasks are automatically routed
to the appropriate inference architecture and mode.
"""

import sys
sys.path.insert(0, '/home/claude')

from inference_architectures.core import (
    ReasoningRouter,
    ModeClassifier,
    InferenceMode,
    Architecture,
)


def mock_llm_call(prompt: str) -> str:
    """
    Mock LLM for demonstration purposes.
    In production, replace with actual API call.
    """
    # Simple response generation for demo
    if "abductive" in prompt.lower() or "hypothesis" in prompt.lower():
        return """HYPOTHESIS: The symptoms suggest hypothyroidism - an underactive thyroid gland that fails to produce sufficient thyroid hormones.
CONFIDENCE: 0.85
REASONING: Fatigue, weight gain, and cold intolerance are classic signs of hypothyroidism, as reduced thyroid hormone slows metabolism."""
    
    elif "deductive" in prompt.lower() or "must follow" in prompt.lower():
        return """CONCLUSION: All whales are warm-blooded.
DERIVATION: From premise 1 (all mammals are warm-blooded) and premise 2 (all whales are mammals), by transitivity, all whales must be warm-blooded.
CERTAINTY: certain"""
    
    elif "inductive" in prompt.lower() or "pattern" in prompt.lower():
        return """PATTERN: Successful B2B startups simplify complex workflows into intuitive interfaces.
CONFIRMING_EVIDENCE: Stripe simplified payments, Slack simplified communication, Figma simplified design collaboration.
POTENTIAL_COUNTEREXAMPLES: Some successful startups (e.g., Palantir) embrace complexity for power users.
SCOPE: Applies primarily to horizontal B2B SaaS targeting broad markets.
CONFIDENCE: 0.75"""
    
    elif "contrastive" in prompt.lower() or "difference" in prompt.lower():
        return """CONTRAST: Traditional programming specifies explicit rules; ML discovers implicit patterns from data.
INSIGHT_ABOUT_A: Traditional programming is transparent but requires full problem specification.
INSIGHT_ABOUT_B: ML can handle problems too complex to specify but operates as a black box.
SYNTHESIZED_UNDERSTANDING: The choice between them depends on whether the problem's rules can be articulated explicitly.
CONFIDENCE: 0.8"""
    
    elif "analogical" in prompt.lower() or "corresponds" in prompt.lower():
        return """CORRESPONDENCES: Central body (sun/nucleus), orbiting bodies (planets/electrons), gravitational/electromagnetic forces.
DIFFERENCES: Quantum effects dominate atoms; deterministic mechanics dominate solar systems. Scale differs by 10^23.
TRANSFERRED_INSIGHT: Both systems involve attraction to a central mass with bodies in stable orbital configurations.
LIMITATIONS: The Bohr model breaks down - electrons don't orbit in classical paths; quantum probability clouds apply.
CONFIDENCE: 0.65"""
    
    elif "syntactic" in prompt.lower() or "organized" in prompt.lower():
        return """ORGANIZATION: Subject-Verb-Object structure. Subject: "The quick brown fox" (NP with determiner + adjectives + noun). Verb: "jumps". Prepositional phrase: "over the lazy dog".
RULES_IDENTIFIED: English SVO word order, adjectives precede nouns, prepositional phrases follow verbs.
VALIDITY: Grammatically valid English sentence following standard syntactic rules."""
    
    elif "qualitative" in prompt.lower() or "intimate" in prompt.lower():
        return """INTUITION: There's a tension between ambition and groundedness that suggests high potential but execution risk.
FELT_SENSE: The combination of 'passionate' and 'underfunded' creates an urgency that could drive either breakthrough or burnout.
CONFIDENCE: 0.6"""
    
    elif "equilibrium" in prompt.lower() or "converge" in prompt.lower():
        return """EQUILIBRIUM: The platform converges toward engagement-maximizing content, even if lower quality.
STABILITY: stable
PATH: Initial quality focus erodes as algorithmic optimization rewards engagement over substance, users adapt to expect sensationalism, and quality contributors leave.
CONFIDENCE: 0.7"""
    
    elif "indexical" in prompt.lower() or "trace" in prompt.lower():
        return """SIGN_DESCRIPTION: Server 500 errors spike consistently at 3am with no scheduled tasks.
INFERRED_CAUSE: Likely a time-based external dependency failure - possibly a third-party API, SSL certificate renewal, or timezone-related cron job misconfiguration.
INTERPRETIVE_LAW: Consistent timing without scheduled triggers suggests external time-dependent factor.
OBSERVER_EFFECTS: Knowing there are no internal scheduled jobs narrows causes to external factors.
CONFIDENCE: 0.7"""
    
    else:
        return """HYPOTHESIS: Unable to determine specific response format.
CONFIDENCE: 0.5
REASONING: Generic response."""


def demo_classification():
    """Demonstrate task classification."""
    print("=" * 60)
    print("DEMO 1: Task Classification")
    print("=" * 60)
    
    classifier = ModeClassifier()
    
    test_tasks = [
        "What would explain the sudden drop in user engagement?",
        "Given that all birds have feathers and all penguins are birds, what must follow?",
        "What pattern emerges from these customer complaints?",
        "What corresponds between a computer network and the nervous system?",
        "What does the difference between synchronous and asynchronous code reveal?",
        "How is this JSON structure organized?",
        "Where does this negotiation process converge?",
        "The logs show memory usage climbing steadily. What does this trace indicate?",
    ]
    
    for task in test_tasks:
        result = classifier.classify(task)
        print(f"\nTask: {task[:60]}...")
        print(f"  Mode: {result.mode.name}")
        print(f"  Architecture: {result.architecture.name}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Signals: {', '.join(result.signals[:2])}")


def demo_reasoning():
    """Demonstrate full reasoning pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 2: Full Reasoning Pipeline")
    print("=" * 60)
    
    router = ReasoningRouter(llm_call=mock_llm_call)
    
    # Test cases for each architecture
    test_cases = [
        {
            "name": "Strange Attractor - Abduction",
            "task": "The patient presents with fatigue, weight gain, and cold intolerance. What would explain these symptoms?",
            "inputs": {"context": "Medical diagnosis scenario"}
        },
        {
            "name": "Code Duality - Deduction",
            "task": "Given: All mammals are warm-blooded. All whales are mammals. What must follow?",
            "inputs": {"premises": ["All mammals are warm-blooded", "All whales are mammals"]}
        },
        {
            "name": "Tensegrity - Induction",
            "task": "What pattern holds across these successful startups?",
            "inputs": {
                "instances": [
                    "Stripe: solved payments complexity with simple API",
                    "Slack: replaced email for team communication",
                    "Figma: made design collaborative in browser"
                ],
                "domain": "B2B software startups"
            }
        },
        {
            "name": "Tensegrity - Contrastive",
            "task": "What does the difference between machine learning and traditional programming reveal?",
            "inputs": {
                "concept_a": "Traditional programming: explicit rules written by humans",
                "concept_b": "Machine learning: patterns learned from data"
            }
        },
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Task: {case['task'][:70]}...")
        
        result = router.reason(
            task=case['task'],
            inputs=case['inputs'],
            max_iterations=5
        )
        
        print(f"\nClassification:")
        print(f"  Mode: {result['classification']['mode']}")
        print(f"  Architecture: {result['classification']['architecture']}")
        print(f"  Confidence: {result['classification']['confidence']:.2f}")
        
        print(f"\nResult:")
        print(f"  Conclusion: {result['result']['conclusion'][:100]}...")
        print(f"  Iterations: {result['metadata']['iterations']}")
        print(f"  Converged: {result['metadata']['converged']}")
        print(f"  Random calls: {result['metadata']['random_calls']}")


def demo_forced_mode():
    """Demonstrate forcing a specific inference mode."""
    print("\n" + "=" * 60)
    print("DEMO 3: Forced Mode Reasoning")
    print("=" * 60)
    
    router = ReasoningRouter(llm_call=mock_llm_call)
    
    task = "Analyze this market situation"
    inputs = {
        "observation": "Tech stocks dropped 5% despite positive earnings reports",
        "context": "Q3 2024 earnings season"
    }
    
    # Run the same task through different modes
    modes_to_test = [
        InferenceMode.ABDUCTION,
        InferenceMode.EQUILIBRIUM,
        InferenceMode.CONTRASTIVE,
    ]
    
    for mode in modes_to_test:
        result = router.reason(
            task=task,
            inputs=inputs,
            force_mode=mode,
            max_iterations=3
        )
        
        print(f"\nForced Mode: {mode.name}")
        print(f"  Architecture: {result['classification']['architecture']}")
        print(f"  Random calls: {result['metadata']['random_calls']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("MULTI-MODAL INFERENCE SYSTEM DEMONSTRATION")
    print("Based on the Intuition Machine 12-Mode Framework")
    print("=" * 60)
    
    demo_classification()
    demo_reasoning()
    demo_forced_mode()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Connect to a real LLM API (replace mock_llm_call)")
    print("2. Connect to QRNG hardware (replace mock in RandomnessProvider)")
    print("3. Run experiments with InferenceExperiment class")
    print("4. Analyze results for entropy sensitivity patterns")


if __name__ == "__main__":
    main()
