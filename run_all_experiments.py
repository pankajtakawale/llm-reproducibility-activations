#!/usr/bin/env python3
"""
Run all activation function experiments in sequence.
Safe for background execution - won't hang on plots.
Auto-detects GPU/CPU and uses appropriate configuration.
Supports multiple model architectures.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - prevents hanging

import torch
import numpy as np
from config import Config
from config_cpu import ConfigCPU
from train import run_experiment
from model_factories import get_model_factory, list_available_models
from plot_utils import plot_training_curves, plot_reproducibility_metrics, plot_summary_metrics
import time
import json
import argparse


def run_all_experiments(models=None, activations=None, config=None):
    """
    Run experiments for multiple models and activations.
    
    Args:
        models: List of model names (default: ['charlm'])
        activations: List of activation names (default: all 5)
        config: Configuration object (default: auto-detect GPU/CPU)
    
    Returns:
        dict: All experiment results
    """
    # Defaults
    if models is None:
        models = ['charlm']
    
    if activations is None:
        activations = ['smelu_05', 'smelu_1', 'relu', 'gelu', 'swish']
    
    # Auto-detect device and use appropriate config if not provided
    if config is None:
        if torch.cuda.is_available():
            config = Config()
            config.device = 'cuda'
            device_name = torch.cuda.get_device_name(0)
            print(f'🚀 GPU detected: {device_name}')
            print(f'   Using full model config ({config.n_layer} layers, {config.n_embd} hidden, {config.max_iters} iters)')
        else:
            config = ConfigCPU()
            config.device = 'cpu'
            print(f'💻 No GPU detected, using CPU-optimized config')
            print(f'   Using lightweight model (2 layers, 128 hidden, 100 iters)')
    
    print('='*60)
    print('RUNNING MULTI-MODEL EXPERIMENTS')
    print('='*60)
    print(f'Device: {config.device}')
    print(f'Models: {", ".join(models)} ({len(models)} total)')
    print(f'Activations: {", ".join(activations)} ({len(activations)} total)')
    print(f'Iterations: {config.max_iters}, Trials: {config.trials_per_activation}')
    print(f'Total experiments: {len(models)} models × {len(activations)} activations = {len(models) * len(activations)}')
    print('='*60)
    
    all_results = {}
    total_start = time.time()
    
    total_experiments = len(models) * len(activations)
    current_experiment = 0
    
    for model_idx, model_name in enumerate(models, 1):
        print(f'\n{"#"*60}')
        print(f'# MODEL {model_idx}/{len(models)}: {model_name.upper()}')
        print(f'{"#"*60}')
        
        # Get model factory
        try:
            model_factory = get_model_factory(model_name)
        except ValueError as e:
            print(f'❌ Error: {e}')
            print(f'   Available models: {", ".join(list_available_models())}')
            continue
        
        model_results = {}
        
        for act_idx, activation in enumerate(activations, 1):
            current_experiment += 1
            print(f'\n[{current_experiment}/{total_experiments}] {model_name} with {activation}...')
            start = time.time()
            
            try:
                # Run experiment with model name for file naming
                results, trained_models, tokenizer = run_experiment(
                    config, 
                    activation,
                    model_name=model_name,
                    model_factory=model_factory
                )
                model_results[activation] = results
                
                # Generate plots (skip if insufficient data)
                # Filename format: {model}_{activation}_{device}
                plot_prefix = f'{model_name}_{activation}_{config.device}'
                try:
                    plot_training_curves(results, plot_prefix, config.plots_dir)
                except Exception as e:
                    print(f'  Warning: Could not generate training curves: {e}')
                
                try:
                    # Only plot reproducibility if we have metrics (requires 2+ trials)
                    if results.get('reproducibility_metrics') and len(results['reproducibility_metrics']) > 0:
                        plot_reproducibility_metrics(results, plot_prefix, config.plots_dir)
                    else:
                        print(f'  Note: Skipping reproducibility plot (need 2+ trials)')
                except Exception as e:
                    print(f'  Warning: Could not generate reproducibility plot: {e}')
                
                try:
                    plot_summary_metrics(results, plot_prefix, config.plots_dir)
                except Exception as e:
                    print(f'  Warning: Could not generate summary plot: {e}')
                
                elapsed = time.time() - start
                print(f'✓ {model_name}/{activation} completed in {elapsed:.1f}s')
                print(f'  Val loss: {results["avg_val_loss"]:.4f}')
                if 'avg_relative_pd' in results and not np.isnan(results['avg_relative_pd']):
                    print(f'  Relative PD: {results["avg_relative_pd"]:.6f}')
                else:
                    print(f'  Relative PD: N/A (need 2+ trials for comparison)')

                
            except Exception as e:
                print(f'❌ Error running {model_name}/{activation}: {e}')
                import traceback
                traceback.print_exc()
                continue
        
        all_results[model_name] = model_results
        
        # Model summary
        if model_results:
            print(f'\n{"="*60}')
            print(f'SUMMARY: {model_name}')
            print(f'{"="*60}')
            for act, res in sorted(model_results.items()):
                print(f'{act:<12} - Loss: {res["avg_val_loss"]:.4f}, PD: {res["avg_relative_pd"]:.6f}')
    
    total_elapsed = time.time() - total_start
    
    # Overall summary
    print(f'\n{"="*60}')
    print('ALL EXPERIMENTS COMPLETED')
    print(f'{"="*60}')
    print(f'Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)')
    print(f'Models: {len(all_results)}')
    print(f'Total experiments: {sum(len(r) for r in all_results.values())}')
    
    # Save summary
    summary = {
        'device': config.device,
        'total_time': total_elapsed,
        'models': list(all_results.keys()),
        'activations': activations,
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
    print(f'Device used: {config.device.upper()}')
    print(f'Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)')
    print(f'\nResults saved to: results/all_experiments_summary.json')
    print(f'Individual results: results/{{model}}-{{activation}}-{{timestamp}}.json')
    suffix = 'gpu' if config.device == 'cuda' else 'cpu'
    print(f'Plots saved to: plots/*_{suffix}_*.png')
    print(f'\nRun "python process_results.py" to generate comprehensive plots and summary!')
    
    return all_results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run multi-model activation experiments'
    )
    parser.add_argument('--models', nargs='+', default=['charlm'],
                       help='Model names to train (default: charlm). Use "all" for all available models')
    parser.add_argument('--activations', nargs='+',
                       default=['smelu_05', 'smelu_1', 'relu', 'gelu', 'swish'],
                       help='Activation functions to test')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        print('Available models:')
        for model in list_available_models():
            print(f'  - {model}')
        return
    
    # Handle "all" models
    if 'all' in args.models:
        args.models = list_available_models()
        print(f'Running all {len(args.models)} models: {", ".join(args.models)}')
    
    # Run experiments
    run_all_experiments(models=args.models, activations=args.activations)


if __name__ == '__main__':
    main()
