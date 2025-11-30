#!/usr/bin/env python3
"""
CPU-optimized quick test script.
Run with: python test_cpu.py
"""
from config_cpu import ConfigCPU
from train import run_experiment
from model_factories import get_model_factory
from plot_utils import plot_training_curves, plot_reproducibility_metrics, plot_summary_metrics
import time

def main():
    config = ConfigCPU()
    print('='*60)
    print('CPU-OPTIMIZED TEST: SmeLU (β=0.5)')
    print('='*60)
    print(f'Model: {config.n_layer} layers, {config.n_embd} hidden, {config.n_head} heads')
    print(f'Iterations: {config.max_iters}, Trials: {config.trials_per_activation}')
    print(f'Model size: ~430K parameters (vs 10.8M in full config)')
    print('='*60)

    charlm_factory = get_model_factory('charlm')
    start_time = time.time()

    # Run experiment
    results, models, tokenizer = run_experiment(config, 'smelu_05', model_factory=charlm_factory)

    elapsed = time.time() - start_time
    print(f'\n⏱️  Completed in: {elapsed:.1f}s ({elapsed/60:.2f} min)')

    # Generate plots
    print('\n📊 Generating plots...')
    plot_training_curves(results, 'smelu_05_cpu', config.plots_dir)
    plot_reproducibility_metrics(results, 'smelu_05_cpu', config.plots_dir)
    plot_summary_metrics(results, 'smelu_05_cpu', config.plots_dir)

    print('\n✅ Test complete!')
    print(f'📄 Results: results/smelu_05_results.json')
    print(f'📊 Plots: plots/smelu_05_cpu_*.png')
    print(f'\n📊 Performance:')
    print(f'  Avg val loss: {results["avg_val_loss"]:.4f}')
    print(f'  Avg relative PD: {results["avg_relative_pd"]:.6f}')
    print(f'  Training time per trial: ~{results["avg_training_time"]:.1f}s')
    print(f'\nEstimate for all 5 activations (100 iters): ~{(elapsed * 5 / 60):.1f} minutes')
    print(f'Estimate for full experiment (5000 iters): ~{(elapsed * 5 * 50 / 60):.0f} minutes')

if __name__ == '__main__':
    main()
