"""
Enhanced training script that tracks reproducibility metrics at multiple checkpoints.
Allows visualization of how Relative PD evolves during training.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from train import train_model, calculate_reproducibility_metrics, calculate_accuracy
from prepare_data import prepare_shakespeare_data
from tokenizer import CharacterTokenizer
import time


def train_with_checkpoints(config, activation_name: str, trial_id: int, 
                          model_factory, tokenizer, data,
                          checkpoint_iters: List[int]) -> Dict:
    """
    Train model and evaluate reproducibility at specific iteration checkpoints.
    
    Args:
        config: Configuration object
        activation_name: Name of activation function
        trial_id: Trial number
        model_factory: Factory function to create model
        tokenizer: Tokenizer instance
        data: Training data
        checkpoint_iters: List of iterations to evaluate at (e.g., [100, 500, 1000, 5000])
    
    Returns:
        Dictionary with training history and checkpoint metrics
    """
    print(f'\n{"="*60}')
    print(f'Training Trial {trial_id} with Checkpoints')
    print(f'{"="*60}')
    
    # Initialize model
    model = model_factory(config, activation_name)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training state
    checkpoint_models = {}
    checkpoint_metrics = []
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    start_time = time.time()
    
    for iter_num in range(config.max_iters):
        # Get batch
        idx = torch.randint(len(data['train']) - config.block_size, (config.batch_size,))
        x = torch.stack([data['train'][i:i+config.block_size] for i in idx]).to(config.device)
        y = torch.stack([data['train'][i+1:i+config.block_size+1] for i in idx]).to(config.device)
        
        # Forward pass
        logits = model(x)
        B, T, C = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), y.view(B*T))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate at intervals
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            # Calculate losses
            with torch.no_grad():
                model.eval()
                
                # Train loss
                train_loss = loss.item()
                
                # Validation loss
                idx = torch.randint(len(data['val']) - config.block_size, (config.batch_size,))
                x_val = torch.stack([data['val'][i:i+config.block_size] for i in idx]).to(config.device)
                y_val = torch.stack([data['val'][i+1:i+config.block_size+1] for i in idx]).to(config.device)
                logits_val = model(x_val)
                val_loss = torch.nn.functional.cross_entropy(
                    logits_val.view(-1, logits_val.size(-1)), 
                    y_val.view(-1)
                ).item()
                
                # Accuracy
                train_acc = calculate_accuracy(model, data['train'], config, num_samples=1000)
                val_acc = calculate_accuracy(model, data['val'], config, num_samples=1000)
                
                model.train()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            elapsed = time.time() - start_time
            print(f'Step {iter_num:5d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Time: {elapsed:.1f}s')
        
        # Save checkpoint if at checkpoint iteration
        if (iter_num + 1) in checkpoint_iters:
            checkpoint_models[iter_num + 1] = model.state_dict().copy()
            print(f'  ✓ Saved checkpoint at iteration {iter_num + 1}')
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f}s')
    print(f'Final train loss: {train_losses[-1]:.4f}')
    print(f'Final val loss: {val_losses[-1]:.4f}')
    print(f'Train accuracy: {train_accuracies[-1]:.2f}%')
    print(f'Val accuracy: {val_accuracies[-1]:.2f}%')
    
    return {
        'trial_id': trial_id,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'training_time': training_time,
        'checkpoint_models': checkpoint_models,
        'final_model': model.state_dict()
    }


def run_experiment_with_checkpoints(config, activation_name: str, model_factory,
                                   checkpoint_iters: List[int] = None) -> Dict:
    """
    Run complete experiment with multiple trials and checkpoint-based reproducibility tracking.
    
    Args:
        config: Configuration object
        activation_name: Name of activation function
        model_factory: Factory function to create model
        checkpoint_iters: Iterations at which to evaluate reproducibility
    
    Returns:
        Dictionary with complete experiment results including checkpoint metrics
    """
    if checkpoint_iters is None:
        # Default checkpoints: 20%, 40%, 60%, 80%, 100% of training
        checkpoint_iters = [
            int(config.max_iters * 0.2),
            int(config.max_iters * 0.4),
            int(config.max_iters * 0.6),
            int(config.max_iters * 0.8),
            config.max_iters
        ]
    
    print(f'\n{"#"*60}')
    print(f'# EXPERIMENT: {activation_name.upper()} WITH CHECKPOINTS')
    print(f'{"#"*60}')
    print(f'Checkpoint iterations: {checkpoint_iters}')
    
    # Prepare data
    data, tokenizer = prepare_shakespeare_data(config)
    
    # Train multiple trials
    trials_data = []
    for trial_id in range(1, config.trials_per_activation + 1):
        # Set seed for this trial
        seed = config.seed_base + trial_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        trial_data = train_with_checkpoints(
            config, activation_name, trial_id,
            model_factory, tokenizer, data,
            checkpoint_iters
        )
        trials_data.append(trial_data)
    
    # Calculate reproducibility at each checkpoint
    print(f'\n{"="*60}')
    print('Calculating Reproducibility Metrics at Checkpoints')
    print(f'{"="*60}')
    
    checkpoint_reproducibility = {}
    
    for checkpoint_iter in checkpoint_iters:
        print(f'\nCheckpoint: Iteration {checkpoint_iter}')
        
        # Load models at this checkpoint
        models_at_checkpoint = []
        for trial_data in trials_data:
            model = model_factory(config, activation_name)
            if checkpoint_iter in trial_data['checkpoint_models']:
                model.load_state_dict(trial_data['checkpoint_models'][checkpoint_iter])
            elif checkpoint_iter == config.max_iters:
                model.load_state_dict(trial_data['final_model'])
            model = model.to(config.device)
            model.eval()
            models_at_checkpoint.append(model)
        
        # Calculate reproducibility between model pairs
        pd_values = []
        for i in range(len(models_at_checkpoint)):
            for j in range(i + 1, len(models_at_checkpoint)):
                metrics = calculate_reproducibility_metrics(
                    models_at_checkpoint[i],
                    models_at_checkpoint[j],
                    data['val'],
                    config
                )
                pd_values.append(metrics['relative_pd'])
                print(f'  Model {i+1} vs {j+1}: Relative PD = {metrics["relative_pd"]:.6f}')
        
        avg_pd = np.mean(pd_values)
        checkpoint_reproducibility[checkpoint_iter] = {
            'iteration': checkpoint_iter,
            'relative_pd_values': pd_values,
            'avg_relative_pd': avg_pd,
            'std_relative_pd': np.std(pd_values)
        }
        print(f'  Average Relative PD: {avg_pd:.6f} ± {np.std(pd_values):.6f}')
    
    # Compile results
    results = {
        'activation': activation_name,
        'checkpoint_iters': checkpoint_iters,
        'trials': [],
        'checkpoint_reproducibility': checkpoint_reproducibility,
        'final_metrics': {}
    }
    
    # Add trial summaries (without full checkpoint models to save space)
    for trial_data in trials_data:
        results['trials'].append({
            'trial_id': trial_data['trial_id'],
            'train_losses': trial_data['train_losses'],
            'val_losses': trial_data['val_losses'],
            'train_accuracies': trial_data['train_accuracies'],
            'val_accuracies': trial_data['val_accuracies'],
            'training_time': trial_data['training_time']
        })
    
    # Calculate final averages
    final_val_losses = [t['val_losses'][-1] for t in results['trials']]
    final_val_accs = [t['val_accuracies'][-1] for t in results['trials']]
    
    results['final_metrics'] = {
        'avg_val_loss': np.mean(final_val_losses),
        'std_val_loss': np.std(final_val_losses),
        'avg_val_accuracy': np.mean(final_val_accs),
        'std_val_accuracy': np.std(final_val_accs),
        'final_relative_pd': checkpoint_reproducibility[config.max_iters]['avg_relative_pd']
    }
    
    # Save results
    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'{activation_name}_checkpoint_results.json'
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, indent=2, fp=f)
    
    print(f'\n✓ Results saved to {results_file}')
    
    return results


if __name__ == '__main__':
    from config_cpu import ConfigCPU
    from model_factories import get_model_factory
    
    # Example usage
    config = ConfigCPU()
    model_factory = get_model_factory('charlm')
    
    # Run experiment with checkpoints
    checkpoint_iters = [20, 40, 60, 80, 100]  # For 100-iteration training
    
    results = run_experiment_with_checkpoints(
        config, 
        'smelu_05',
        model_factory,
        checkpoint_iters
    )
    
    print('\n📊 Checkpoint reproducibility evolution:')
    for iter_num in checkpoint_iters:
        pd = results['checkpoint_reproducibility'][iter_num]['avg_relative_pd']
        print(f'  Iteration {iter_num}: Relative PD = {pd:.6f}')
