"""
Standalone data processing script to read all result files and generate plots/tables.
Reads {model}-{activation}-{timestamp}.json files from results/ directory.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse
import re


def parse_filename(filename):
    """
    Parse result filename to extract model, activation, and timestamp.
    
    Format: {model}-{activation}-{timestamp}.json
    Example: charlm-smelu_05-20251206_143022.json
    
    Returns:
        dict with 'model', 'activation', 'timestamp' keys, or None if parsing fails
    """
    pattern = r'^(.+?)-(.+?)-(\d{8}_\d{6})\.json$'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'model': match.group(1),
            'activation': match.group(2),
            'timestamp': match.group(3)
        }
    return None


def load_all_results(results_dir='results', models=None, activations=None):
    """
    Load all result files from results directory.
    
    Args:
        results_dir: Directory containing result JSON files
        models: List of model names to filter (None = all)
        activations: List of activation names to filter (None = all)
    
    Returns:
        List of dicts, each containing parsed data and metadata
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found!")
        return []
    
    all_results = []
    
    for json_file in results_path.glob('*.json'):
        # Parse filename
        parsed = parse_filename(json_file.name)
        
        if parsed is None:
            print(f"Skipping file with unexpected format: {json_file.name}")
            continue
        
        # Apply filters
        if models and parsed['model'] not in models:
            continue
        if activations and parsed['activation'] not in activations:
            continue
        
        # Load data
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Add metadata
            data['filename'] = json_file.name
            data['filepath'] = str(json_file)
            data['parsed_model'] = parsed['model']
            data['parsed_activation'] = parsed['activation']
            data['parsed_timestamp'] = parsed['timestamp']
            
            all_results.append(data)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    print(f"Loaded {len(all_results)} result files")
    return all_results


def aggregate_by_model_activation(results):
    """
    Aggregate results by model and activation.
    
    Returns:
        dict: {model_name: {activation_name: [list of results]}}
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        model = result.get('model_name', result.get('parsed_model', 'unknown'))
        activation = result.get('activation', result.get('parsed_activation', 'unknown'))
        aggregated[model][activation].append(result)
    
    return aggregated


def create_summary_table(aggregated_data, output_file='results/summary.txt'):
    """
    Create a comprehensive summary table of all results.
    
    Args:
        aggregated_data: Output from aggregate_by_model_activation()
        output_file: Path to save the summary text
    """
    lines = []
    lines.append("="*100)
    lines.append("COMPREHENSIVE EXPERIMENT SUMMARY")
    lines.append("="*100)
    lines.append("")
    
    for model_name, activations in sorted(aggregated_data.items()):
        lines.append(f"\nModel: {model_name.upper()}")
        lines.append("-"*100)
        lines.append(f"{'Activation':<15} {'Val Loss':<15} {'Val Acc (%)':<15} {'Relative PD':<15} {'Time (s)':<15} {'Trials':<10}")
        lines.append("-"*100)
        
        for activation, results in sorted(activations.items()):
            # Average across all runs for this model-activation pair
            avg_val_loss = np.mean([r['avg_val_loss'] for r in results])
            avg_val_acc = np.mean([r['avg_val_accuracy'] for r in results])
            avg_rel_pd = np.mean([r['avg_relative_pd'] for r in results])
            avg_time = np.mean([r['avg_training_time'] for r in results])
            
            # Count total trials across all runs
            total_trials = sum(len(r.get('trials', [])) for r in results)
            
            lines.append(f"{activation:<15} {avg_val_loss:<15.4f} {avg_val_acc:<15.2f} {avg_rel_pd:<15.6f} {avg_time:<15.1f} {total_trials:<10}")
        
        lines.append("")
    
    # Overall best performers
    lines.append("\n" + "="*100)
    lines.append("BEST PERFORMERS")
    lines.append("="*100)
    
    # Find best for each metric
    all_flat = [(model, act, r) for model, acts in aggregated_data.items() 
                for act, results in acts.items() for r in results]
    
    if all_flat:
        # Best accuracy
        best_acc = max(all_flat, key=lambda x: x[2]['avg_val_accuracy'])
        lines.append(f"\nBest Accuracy: {best_acc[0]} with {best_acc[1]}")
        lines.append(f"  Val Accuracy: {best_acc[2]['avg_val_accuracy']:.2f}%")
        
        # Best reproducibility (lowest PD)
        best_repro = min(all_flat, key=lambda x: x[2]['avg_relative_pd'])
        lines.append(f"\nBest Reproducibility: {best_repro[0]} with {best_repro[1]}")
        lines.append(f"  Relative PD: {best_repro[2]['avg_relative_pd']:.6f}")
        
        # Best loss
        best_loss = min(all_flat, key=lambda x: x[2]['avg_val_loss'])
        lines.append(f"\nBest Loss: {best_loss[0]} with {best_loss[1]}")
        lines.append(f"  Val Loss: {best_loss[2]['avg_val_loss']:.4f}")
        
        # Fastest training
        fastest = min(all_flat, key=lambda x: x[2]['avg_training_time'])
        lines.append(f"\nFastest Training: {fastest[0]} with {fastest[1]}")
        lines.append(f"  Training Time: {fastest[2]['avg_training_time']:.1f}s")
    
    lines.append("\n" + "="*100)
    
    # Print and save
    summary_text = "\n".join(lines)
    print(summary_text)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(summary_text)
    
    print(f"\nSummary saved to {output_file}")


def plot_accuracy_comparison(aggregated_data, output_dir='plots'):
    """Plot accuracy comparison across models and activations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models = sorted(aggregated_data.keys())
    
    for model_name in models:
        activations = aggregated_data[model_name]
        
        act_names = sorted(activations.keys())
        val_accs = []
        val_acc_stds = []
        
        for act in act_names:
            results = activations[act]
            accs = [r['avg_val_accuracy'] for r in results]
            val_accs.append(np.mean(accs))
            val_acc_stds.append(np.std(accs) if len(accs) > 1 else 0)
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(act_names))
        plt.bar(x, val_accs, yerr=val_acc_stds, capsize=5, alpha=0.7)
        plt.xlabel('Activation Function')
        plt.ylabel('Validation Accuracy (%)')
        plt.title(f'Accuracy Comparison: {model_name}')
        plt.xticks(x, act_names, rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plot_file = output_path / f'{model_name}_accuracy.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_file}")


def plot_reproducibility_comparison(aggregated_data, output_dir='plots'):
    """Plot reproducibility (Relative PD) comparison across models and activations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models = sorted(aggregated_data.keys())
    
    for model_name in models:
        activations = aggregated_data[model_name]
        
        act_names = sorted(activations.keys())
        rel_pds = []
        rel_pd_stds = []
        
        for act in act_names:
            results = activations[act]
            pds = [r['avg_relative_pd'] for r in results]
            rel_pds.append(np.mean(pds))
            rel_pd_stds.append(np.std(pds) if len(pds) > 1 else 0)
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(act_names))
        plt.bar(x, rel_pds, yerr=rel_pd_stds, capsize=5, alpha=0.7, color='coral')
        plt.xlabel('Activation Function')
        plt.ylabel('Relative Prediction Difference')
        plt.title(f'Reproducibility Comparison: {model_name} (Lower is Better)')
        plt.xticks(x, act_names, rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plot_file = output_path / f'{model_name}_reproducibility.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_file}")


def plot_multi_model_comparison(aggregated_data, output_dir='plots'):
    """Create grouped bar chart comparing all models across activations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all activations across all models
    all_activations = set()
    for activations in aggregated_data.values():
        all_activations.update(activations.keys())
    
    act_names = sorted(all_activations)
    models = sorted(aggregated_data.keys())
    
    if len(models) <= 1:
        print("Need at least 2 models for multi-model comparison")
        return
    
    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(act_names))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        accs = []
        for act in act_names:
            if act in aggregated_data[model]:
                results = aggregated_data[model][act]
                avg_acc = np.mean([r['avg_val_accuracy'] for r in results])
                accs.append(avg_acc)
            else:
                accs.append(0)
        
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Multi-Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(act_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_path / 'multi_model_accuracy.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file}")
    
    # Reproducibility comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, model in enumerate(models):
        pds = []
        for act in act_names:
            if act in aggregated_data[model]:
                results = aggregated_data[model][act]
                avg_pd = np.mean([r['avg_relative_pd'] for r in results])
                pds.append(avg_pd)
            else:
                pds.append(0)
        
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, pds, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Relative Prediction Difference')
    ax.set_title('Multi-Model Reproducibility Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(act_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_path / 'multi_model_reproducibility.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file}")


def plot_training_curves(aggregated_data, output_dir='plots'):
    """Plot training curves for each model-activation pair."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, activations in aggregated_data.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for activation, results in sorted(activations.items()):
            # Average loss history across all runs
            all_train_losses = []
            all_val_losses = []
            
            for result in results:
                if 'trials' in result:
                    for trial in result['trials']:
                        if 'train_loss_history' in trial:
                            all_train_losses.append(trial['train_loss_history'])
                        if 'val_loss_history' in trial:
                            all_val_losses.append(trial['val_loss_history'])
            
            if all_train_losses:
                # Average across trials
                min_len = min(len(l) for l in all_train_losses)
                train_avg = np.mean([l[:min_len] for l in all_train_losses], axis=0)
                axes[0].plot(train_avg, label=activation, alpha=0.7)
            
            if all_val_losses:
                min_len = min(len(l) for l in all_val_losses)
                val_avg = np.mean([l[:min_len] for l in all_val_losses], axis=0)
                axes[1].plot(val_avg, label=activation, alpha=0.7)
        
        axes[0].set_xlabel('Evaluation Step')
        axes[0].set_ylabel('Train Loss')
        axes[0].set_title(f'{model_name}: Training Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].set_xlabel('Evaluation Step')
        axes[1].set_ylabel('Validation Loss')
        axes[1].set_title(f'{model_name}: Validation Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_path / f'{model_name}_training_curves.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_file}")


def plot_accuracy_vs_reproducibility(aggregated_data, output_dir='plots'):
    """Scatter plot: accuracy vs reproducibility trade-off."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(aggregated_data)))
    
    for (model_name, activations), color in zip(sorted(aggregated_data.items()), colors):
        accs = []
        pds = []
        labels = []
        
        for activation, results in sorted(activations.items()):
            avg_acc = np.mean([r['avg_val_accuracy'] for r in results])
            avg_pd = np.mean([r['avg_relative_pd'] for r in results])
            accs.append(avg_acc)
            pds.append(avg_pd)
            labels.append(activation)
        
        ax.scatter(pds, accs, s=100, alpha=0.6, c=[color], label=model_name)
        
        # Annotate points with activation names
        for i, label in enumerate(labels):
            ax.annotate(label, (pds[i], accs[i]), fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Relative Prediction Difference (Lower is Better)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Accuracy vs Reproducibility Trade-off')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_path / 'accuracy_vs_reproducibility.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Process experiment results and generate plots/tables'
    )
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output-dir', default='plots',
                       help='Directory to save plots')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Filter by model names (e.g., charlm tinylstm)')
    parser.add_argument('--activations', nargs='+', default=None,
                       help='Filter by activation names (e.g., relu gelu)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots (only create summary)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PROCESSING EXPERIMENT RESULTS")
    print("="*60)
    
    # Load all results
    results = load_all_results(
        results_dir=args.results_dir,
        models=args.models,
        activations=args.activations
    )
    
    if not results:
        print("No results found!")
        return
    
    # Aggregate data
    aggregated = aggregate_by_model_activation(results)
    
    # Create summary table
    create_summary_table(aggregated, output_file=f'{args.results_dir}/summary.txt')
    
    # Generate plots
    if not args.no_plots:
        print(f"\n{'='*60}")
        print("GENERATING PLOTS")
        print(f"{'='*60}")
        
        plot_accuracy_comparison(aggregated, args.output_dir)
        plot_reproducibility_comparison(aggregated, args.output_dir)
        plot_training_curves(aggregated, args.output_dir)
        plot_accuracy_vs_reproducibility(aggregated, args.output_dir)
        
        # Multi-model plots (only if multiple models)
        if len(aggregated) > 1:
            plot_multi_model_comparison(aggregated, args.output_dir)
        
        print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
