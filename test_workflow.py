#!/usr/bin/env python3
"""
Quick test of multi-model experiment workflow.
Tests with 2 models and 2 activations for fast validation.
"""
import matplotlib
matplotlib.use('Agg')

from run_all_experiments import run_all_experiments
from config_cpu import ConfigCPU

# Override iterations for quick test
config = ConfigCPU()
config.max_iters = 20  # Very short training
config.eval_interval = 10
config.trials_per_activation = 2  # 2 trials for valid Relative PD

print("="*60)
print("QUICK TEST: Multi-Model Workflow")
print("="*60)
print("Testing with:")
print("  - 2 models: charlm, tinylstm")
print("  - 2 activations: relu, gelu")
print("  - 20 iterations per trial")
print("  - 2 trials per activation (for valid Relative PD)")
print("="*60)

# Run quick test
results = run_all_experiments(
    models=['charlm', 'tinylstm'],
    activations=['relu', 'gelu'],
    config=config
)

print("\n" + "="*60)
print("TEST COMPLETED!")
print("="*60)
print("\nNext steps:")
print("1. Check results/ directory for {model}-{activation}-{timestamp}.json files")
print("2. Run: python process_results.py")
print("3. Check plots/ directory for generated visualizations")
print("4. Check results/summary.txt for comprehensive analysis")
