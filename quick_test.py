"""
Quick test script for running experiments with reduced iterations.
Use this to verify the pipeline before running full experiments.
"""
from config import Config
from train import run_experiment
from model_factories import get_model_factory
from plot_utils import plot_training_curves, plot_reproducibility_metrics, plot_summary_metrics
import time

# Quick test configuration
config = Config()
config.max_iters = 500  # Reduced from 5000 for quick test (~1 min per trial)
config.trials_per_activation = 2  # Reduced from 3

print('='*60)
print('QUICK TEST: SmeLU (β=0.5) with reduced iterations')
print('='*60)
print(f'Iterations: {config.max_iters}')
print(f'Trials: {config.trials_per_activation}')
print(f'Estimated time: ~2-3 minutes')
print('='*60)
print()

start_time = time.time()

# Get model factory
charlm_factory = get_model_factory('charlm')

# Run one activation as test
results, models, tokenizer = run_experiment(config, 'smelu_05', model_factory=charlm_factory)

elapsed = time.time() - start_time

# Generate plots
print('\n📊 Generating visualizations...')
plot_training_curves(results, 'smelu_05', config.plots_dir)
plot_reproducibility_metrics(results, 'smelu_05', config.plots_dir)
plot_summary_metrics(results, 'smelu_05', config.plots_dir)

print('\n' + '='*60)
print('✅ QUICK TEST COMPLETE!')
print('='*60)
print(f'⏱️  Total time: {elapsed/60:.1f} minutes')
print(f'📄 Results: results/smelu_05_results.json')
print(f'📊 Plots: plots/smelu_05_*.png')
print('\n' + '='*60)
print('NEXT STEPS:')
print('='*60)
print('1. Check the plots in plots/ folder')
print('2. Review metrics in results/smelu_05_results.json')
print('3. If satisfied, run full experiments:')
print('   - Set config.max_iters = 5000')
print('   - Set config.trials_per_activation = 3')
print('   - Estimated time: 1.5-2 hours for all 5 activations')
print('='*60)
