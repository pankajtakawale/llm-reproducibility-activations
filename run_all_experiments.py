#!/usr/bin/env python3
"""
Run all activation function experiments in sequence.
Safe for background execution - won't hang on plots.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - prevents hanging

from config_cpu import ConfigCPU
from train import run_experiment
from model_factories import get_model_factory
from plot_utils import plot_training_curves, plot_reproducibility_metrics, plot_summary_metrics
import time
import json

def main():
    config = ConfigCPU()
    print('='*60)
    print('RUNNING ALL ACTIVATION EXPERIMENTS')
    print('='*60)
    print(f'Model: {config.n_layer} layers, {config.n_embd} hidden, {config.n_head} heads')
    print(f'Iterations: {config.max_iters}, Trials: {config.trials_per_activation}')
    print(f'Activations: smelu_05, smelu_1, relu, gelu, swish')
    print('='*60)

    charlm_factory = get_model_factory('charlm')
    activations = ['smelu_05', 'smelu_1', 'relu', 'gelu', 'swish']
    
    all_results = {}
    total_start = time.time()
    
    for i, activation in enumerate(activations, 1):
        print(f'\n[{i}/5] Running {activation}...')
        start = time.time()
        
        # Run experiment
        results, models, tokenizer = run_experiment(config, activation, model_factory=charlm_factory)
        all_results[activation] = results
        
        # Generate plots
        plot_training_curves(results, f'{activation}_cpu', config.plots_dir)
        plot_reproducibility_metrics(results, f'{activation}_cpu', config.plots_dir)
        plot_summary_metrics(results, f'{activation}_cpu', config.plots_dir)
        
        elapsed = time.time() - start
        print(f'✓ {activation} completed in {elapsed:.1f}s')
        print(f'  Val loss: {results["avg_val_loss"]:.4f}')
        print(f'  Relative PD: {results["avg_relative_pd"]:.6f}')
    
    total_elapsed = time.time() - total_start
    
    # Save summary
    summary = {
        'total_time': total_elapsed,
        'config': {
            'n_layer': config.n_layer,
            'n_embd': config.n_embd,
            'max_iters': config.max_iters,
            'trials': config.trials_per_activation
        },
        'results': all_results
    }
    
    with open('results/all_experiments_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('\n' + '='*60)
    print('ALL EXPERIMENTS COMPLETE!')
    print('='*60)
    print(f'Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)')
    print(f'\nResults saved to: results/all_experiments_summary.json')
    print(f'Plots saved to: plots/*_cpu_*.png')
    
    # Print comparison
    print('\nComparison:')
    print(f"{'Activation':<12} {'Val Loss':<12} {'Relative PD':<15} {'Time (s)':<10}")
    print('-'*60)
    for act, res in all_results.items():
        print(f"{act:<12} {res['avg_val_loss']:<12.4f} {res['avg_relative_pd']:<15.6f} {res['avg_training_time']:<10.1f}")

if __name__ == '__main__':
    main()
