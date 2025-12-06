"""
Training script with reproducibility metrics.
Implements relative prediction difference (PD) calculation following the original project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import json
from pathlib import Path
from datetime import datetime


def set_seed(seed):
    """Set random seeds for reproducibility within a trial."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_batch(data, batch_size, block_size, device):
    """
    Get a random batch of data.
    
    Args:
        data: Encoded text (list of ints)
        batch_size: Number of sequences
        block_size: Length of each sequence
        device: torch device
    
    Returns:
        x: Input sequences (batch_size, block_size)
        y: Target sequences (batch_size, block_size)
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    """Estimate loss on train and val sets."""
    model.eval()
    out = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(data, config.batch_size, config.block_size, config.device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


@torch.no_grad()
def calculate_accuracy(model, data, config, num_samples=1000):
    """
    Calculate top-1 accuracy on a subset of data.
    
    Args:
        model: Trained model
        data: Encoded text
        config: Configuration
        num_samples: Number of sequences to evaluate
    
    Returns:
        Accuracy as a percentage (0-100)
    """
    model.eval()
    correct = 0
    total = 0
    
    for _ in range(num_samples):
        x, y = get_batch(data, 1, config.block_size, config.device)
        logits, _ = model(x)
        
        # Get predictions for the last token
        pred = logits[0, -1, :].argmax()
        target = y[0, -1]
        
        if pred == target:
            correct += 1
        total += 1
    
    accuracy = (correct / total) * 100
    return accuracy


@torch.no_grad()
def get_predictions(model, data, config, num_samples=1000):
    """
    Get model predictions on a subset of data.
    Used for calculating reproducibility metrics.
    
    Args:
        model: Trained model
        data: Encoded text
        config: Configuration
        num_samples: Number of sequences to evaluate
    
    Returns:
        predictions: Array of probability distributions (num_samples, vocab_size)
    """
    model.eval()
    predictions = []
    
    for _ in range(num_samples):
        x, y = get_batch(data, 1, config.block_size, config.device)
        logits, _ = model(x)
        
        # Get probabilities for the last token
        probs = F.softmax(logits[0, -1, :], dim=-1)
        predictions.append(probs.cpu().numpy())
    
    return np.array(predictions)


def calculate_relative_pd(preds1, preds2):
    """
    Calculate relative prediction difference between two models.
    
    Relative PD = mean(|p1 - p2|) / (mean(p1) + mean(p2))
    
    Args:
        preds1: Predictions from model 1 (N, vocab_size)
        preds2: Predictions from model 2 (N, vocab_size)
    
    Returns:
        Relative PD value
    """
    # Absolute difference
    abs_diff = np.abs(preds1 - preds2)
    
    # Mean predictions
    mean_p1 = np.mean(preds1)
    mean_p2 = np.mean(preds2)
    
    # Relative PD
    relative_pd = np.mean(abs_diff) / (mean_p1 + mean_p2)
    
    return relative_pd


def calculate_prediction_differences(preds1, preds2):
    """
    Calculate number of different top-1 predictions.
    
    Args:
        preds1: Predictions from model 1 (N, vocab_size)
        preds2: Predictions from model 2 (N, vocab_size)
    
    Returns:
        Number of different predictions
    """
    # Get top-1 predictions
    top1_preds1 = np.argmax(preds1, axis=1)
    top1_preds2 = np.argmax(preds2, axis=1)
    
    # Count differences
    differences = np.sum(top1_preds1 != top1_preds2)
    
    return differences


def train_model(model, train_data, val_data, config, trial_id):
    """
    Train a single model.
    
    Args:
        model: Model to train
        train_data: Training data (encoded)
        val_data: Validation data (encoded)
        config: Configuration object
        trial_id: Trial identifier
    
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Training Trial {trial_id}")
    print(f"{'='*60}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    
    for iter in range(config.max_iters):
        # Evaluate periodically
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            
            elapsed = time.time() - start_time
            print(f"Step {iter:5d} | Train loss: {losses['train']:.4f} | "
                  f"Val loss: {losses['val']:.4f} | Time: {elapsed:.1f}s")
        
        # Training step
        x, y = get_batch(train_data, config.batch_size, config.block_size, config.device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    total_time = time.time() - start_time
    
    # Final evaluation
    final_losses = estimate_loss(model, train_data, val_data, config)
    
    # Calculate accuracy
    train_accuracy = calculate_accuracy(model, train_data, config)
    val_accuracy = calculate_accuracy(model, val_data, config)
    
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Final train loss: {final_losses['train']:.4f}")
    print(f"Final val loss: {final_losses['val']:.4f}")
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Val accuracy: {val_accuracy:.2f}%")
    
    results = {
        'trial_id': trial_id,
        'train_loss': float(final_losses['train']),
        'val_loss': float(final_losses['val']),
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'training_time': total_time,
        'train_loss_history': [float(x) for x in train_losses],
        'val_loss_history': [float(x) for x in val_losses]
    }
    
    return results


def run_experiment(config, activation_name, model_name='charlm', model_factory=None, data_loader=None, tokenizer_factory=None):
    """
    Run full experiment for one activation function with pluggable components.
    
    Args:
        config: Configuration object
        activation_name: Name of activation function
        model_name: Name of the model architecture (for file naming)
        model_factory: Optional callable that creates a model. 
                      Signature: model_factory(config, activation) -> model
                      If None, defaults to CharLM
        data_loader: Optional callable that loads and prepares data.
                    Signature: data_loader(config) -> (train_data, val_data, tokenizer)
                    If None, defaults to Shakespeare dataset
        tokenizer_factory: Optional callable that creates a tokenizer.
                          Signature: tokenizer_factory(text) -> tokenizer
                          If None, defaults to CharTokenizer
    
    Returns:
        Tuple of (experiment_results dict, models list, tokenizer)
    """
    from prepare_data import load_shakespeare, prepare_data
    from tokenizer import CharTokenizer
    from model import CharLM
    from activations import get_activation
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT: {activation_name.upper()}")
    print(f"{'#'*60}")
    
    # Use default data loader if not provided
    if data_loader is None:
        # Default: Shakespeare dataset
        text = load_shakespeare()
        train_text, val_text = prepare_data(text, config.train_split)
        
        # Use default tokenizer if not provided
        if tokenizer_factory is None:
            tokenizer = CharTokenizer(text)
        else:
            tokenizer = tokenizer_factory(text)
        
        config.vocab_size = len(tokenizer)
        train_data = tokenizer.encode(train_text)
        val_data = tokenizer.encode(val_text)
    else:
        # Custom data loader
        train_data, val_data, tokenizer = data_loader(config)
        config.vocab_size = len(tokenizer)
    
    # Get activation function
    activation = get_activation(activation_name)
    
    # Use default model factory if not provided
    if model_factory is None:
        def default_model_factory(cfg, act):
            return CharLM(
                vocab_size=cfg.vocab_size,
                n_embd=cfg.n_embd,
                n_head=cfg.n_head,
                n_layer=cfg.n_layer,
                block_size=cfg.block_size,
                activation=act,
                dropout=cfg.dropout
            )
        model_factory = default_model_factory
    
    # Train multiple models
    models = []
    results = []
    
    for trial in range(config.trials_per_activation):
        # Set seed for this trial
        seed = config.seed_base + trial
        set_seed(seed)
        
        # Create model using factory
        model = model_factory(config, activation).to(config.device)
        
        # Train
        result = train_model(model, train_data, val_data, config, trial + 1)
        results.append(result)
        models.append(model)
        
        # Save checkpoint if enabled
        if config.save_checkpoints:
            checkpoint_dir = Path(config.checkpoint_dir) / activation_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"trial_{trial + 1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'result': result
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Calculate reproducibility metrics between models
    print(f"\n{'='*60}")
    print("Calculating Reproducibility Metrics")
    print(f"{'='*60}")
    
    reproducibility_metrics = []
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            print(f"Comparing Model {i+1} vs Model {j+1}...")
            
            preds1 = get_predictions(models[i], val_data, config)
            preds2 = get_predictions(models[j], val_data, config)
            
            rel_pd = calculate_relative_pd(preds1, preds2)
            pred_diffs = calculate_prediction_differences(preds1, preds2)
            
            metric = {
                'model_pair': f"{i+1}_vs_{j+1}",
                'relative_pd': float(rel_pd),
                'prediction_differences': int(pred_diffs),
                'total_predictions': len(preds1)
            }
            reproducibility_metrics.append(metric)
            
            print(f"  Relative PD: {rel_pd:.6f}")
            print(f"  Prediction differences: {pred_diffs}/{len(preds1)}")
    
    # Aggregate results
    experiment_results = {
        'model_name': model_name,
        'activation': activation_name,
        'trials': results,
        'reproducibility_metrics': reproducibility_metrics,
        'avg_train_loss': np.mean([r['train_loss'] for r in results]),
        'avg_val_loss': np.mean([r['val_loss'] for r in results]),
        'std_val_loss': np.std([r['val_loss'] for r in results]),
        'avg_train_accuracy': np.mean([r['train_accuracy'] for r in results]),
        'avg_val_accuracy': np.mean([r['val_accuracy'] for r in results]),
        'std_val_accuracy': np.std([r['val_accuracy'] for r in results]),
        'avg_relative_pd': np.mean([m['relative_pd'] for m in reproducibility_metrics]),
        'avg_training_time': np.mean([r['training_time'] for r in results]),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results with new naming format: {model}-{activation}-{timestamp}.json
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f"{model_name}-{activation_name}-{timestamp_str}.json"
    results_path = results_dir / results_filename
    
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment Summary: {model_name} with {activation_name}")
    print(f"{'='*60}")
    print(f"Average val loss: {experiment_results['avg_val_loss']:.4f} ± {experiment_results['std_val_loss']:.4f}")
    print(f"Average val accuracy: {experiment_results['avg_val_accuracy']:.2f}% ± {experiment_results['std_val_accuracy']:.2f}%")
    print(f"Average relative PD: {experiment_results['avg_relative_pd']:.6f}")
    print(f"Average training time: {experiment_results['avg_training_time']:.1f}s")
    print(f"Results saved to {results_path}")
    
    return experiment_results, models, tokenizer


if __name__ == '__main__':
    from config import Config
    
    # Test with one activation
    config = Config()
    results, models, tokenizer = run_experiment(config, 'relu')
