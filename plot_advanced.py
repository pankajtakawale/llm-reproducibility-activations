"""
Advanced plotting utilities for multi-model, multi-activation analysis.
Generates comprehensive visualizations for comparing reproducibility across
different model architectures and activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_multi_model_results(results_dir: str = 'results') -> Dict:
    """
    Load results from multiple models and organize them hierarchically.
    
    Returns:
        {
            'charlm': {
                'smelu_05': {...},
                'relu': {...},
                ...
            },
            'gpt2': {
                'smelu_05': {...},
                ...
            }
        }
    """
    results_path = Path(results_dir)
    multi_model_data = {}
    
    # Look for model-specific subdirectories or files
    # Pattern: results/charlm/smelu_05_results.json or results/charlm_smelu_05_results.json
    
    for result_file in results_path.glob('**/*_results.json'):
        # Skip summary files
        if 'summary' in result_file.name:
            continue
            
        # Try to extract model name from path or filename
        parts = result_file.parts
        if len(parts) > 2 and parts[-2] != 'results':
            model_name = parts[-2]  # results/charlm/smelu_05_results.json
        else:
            # Extract from filename: charlm_smelu_05_results.json
            name_parts = result_file.stem.split('_')
            if len(name_parts) > 2:
                model_name = name_parts[0]
            else:
                model_name = 'charlm'  # default
        
        # Extract activation name
        if 'smelu_05' in result_file.name:
            activation = 'smelu_05'
        elif 'smelu_1' in result_file.name:
            activation = 'smelu_1'
        elif 'relu' in result_file.name:
            activation = 'relu'
        elif 'gelu' in result_file.name:
            activation = 'gelu'
        elif 'swish' in result_file.name:
            activation = 'swish'
        else:
            activation = result_file.stem.replace('_results', '')
        
        # Load data
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Organize hierarchically
        if model_name not in multi_model_data:
            multi_model_data[model_name] = {}
        multi_model_data[model_name][activation] = data
    
    return multi_model_data


def plot_heatmap_reproducibility(multi_model_data: Dict, save_path: str = 'plots/heatmap_reproducibility.png'):
    """
    Heatmap showing Relative PD across Models (rows) × Activations (columns).
    
    Args:
        multi_model_data: Hierarchical dict with models and activations
        save_path: Where to save the plot
    """
    # Extract data
    models = sorted(multi_model_data.keys())
    all_activations = set()
    for model_data in multi_model_data.values():
        all_activations.update(model_data.keys())
    activations = sorted(all_activations)
    
    # Create matrix
    matrix = np.zeros((len(models), len(activations)))
    for i, model in enumerate(models):
        for j, activation in enumerate(activations):
            if activation in multi_model_data[model]:
                data = multi_model_data[model][activation]
                matrix[i, j] = data['avg_relative_pd']
            else:
                matrix[i, j] = np.nan
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Heatmap
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.45, vmax=0.55)
    
    # Set ticks
    ax.set_xticks(np.arange(len(activations)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([act.replace('_', ' ').upper() for act in activations], rotation=45, ha='right')
    ax.set_yticklabels([m.upper() for m in models])
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(activations)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                              ha='center', va='center', color='black', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Activation Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_title('Reproducibility Heatmap: Relative PD by Model & Activation\n(Lower is Better - Darker Red)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Prediction Disagreement', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'📊 Saved heatmap to {save_path}')


def plot_training_evolution(multi_model_data: Dict, metric: str = 'relative_pd',
                            save_path: str = 'plots/training_evolution.png'):
    """
    Line plots showing how metrics evolve during training.
    One subplot per model, multiple lines per activation.
    
    Args:
        multi_model_data: Hierarchical dict with models and activations
        metric: 'relative_pd', 'val_loss', or 'val_accuracy'
        save_path: Where to save the plot
    """
    models = sorted(multi_model_data.keys())
    n_models = len(models)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = multi_model_data[model]
        
        for color_idx, (activation, data) in enumerate(sorted(model_data.items())):
            # Extract iteration-level data if available
            trials = data.get('trials', [])
            if not trials:
                continue
            
            # Get metric history
            if metric == 'relative_pd':
                # For relative PD, we need to calculate at each checkpoint
                # For now, plot final value as horizontal line
                y_val = data['avg_relative_pd']
                iterations = [0, 100]  # Placeholder
                y_vals = [y_val, y_val]
            elif metric == 'val_loss':
                # Use loss history from trials
                iterations = []
                y_vals = []
                for trial in trials:
                    history = trial.get('val_loss_history', [])
                    if history:
                        trial_iters = np.linspace(0, len(history)-1, len(history))
                        iterations.extend(trial_iters)
                        y_vals.extend(history)
            elif metric == 'val_accuracy':
                # Use accuracy if available
                y_val = data.get('avg_val_accuracy', 0)
                iterations = [0, 100]
                y_vals = [y_val, y_val]
            
            # Plot
            label = activation.replace('_', ' ').upper()
            ax.plot(iterations, y_vals, label=label, color=colors[color_idx],
                   linewidth=2, alpha=0.7, marker='o', markersize=4)
        
        # Formatting
        ax.set_xlabel('Training Iteration', fontsize=11, fontweight='bold')
        if idx == 0:
            ylabel = metric.replace('_', ' ').title()
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # Overall title
    metric_title = metric.replace('_', ' ').title()
    fig.suptitle(f'{metric_title} Evolution During Training', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'📊 Saved training evolution to {save_path}')


def plot_accuracy_vs_reproducibility(multi_model_data: Dict,
                                     save_path: str = 'plots/accuracy_vs_reproducibility.png'):
    """
    Scatter plot showing the trade-off between accuracy and reproducibility.
    Each point is a model-activation combination.
    
    Args:
        multi_model_data: Hierarchical dict with models and activations
        save_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define colors and markers
    activation_colors = {
        'smelu_05': '#e74c3c',
        'smelu_1': '#c0392b',
        'relu': '#3498db',
        'gelu': '#2ecc71',
        'swish': '#f39c12'
    }
    
    model_markers = {
        'charlm': 'o',
        'gpt2': 's',
        'nanogpt': '^',
        'babygpt': 'D'
    }
    
    # Collect data
    for model, model_data in multi_model_data.items():
        for activation, data in model_data.items():
            x = data['avg_val_loss']  # Lower is better
            y = data['avg_relative_pd']  # Lower is better
            
            color = activation_colors.get(activation, '#95a5a6')
            marker = model_markers.get(model, 'o')
            
            # Plot point
            ax.scatter(x, y, c=color, marker=marker, s=150, alpha=0.7,
                      edgecolors='black', linewidth=1.5,
                      label=f'{model}-{activation}')
            
            # Add annotation
            label = f'{activation.replace("_", " ")}'
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Validation Loss (Lower = Better Accuracy)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative PD (Lower = Better Reproducibility)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Reproducibility Trade-off\nAcross Models and Activations',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Invert axes to show "better" in top-right
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Add Pareto frontier annotation
    ax.text(0.05, 0.95, '← Better Accuracy', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', style='italic')
    ax.text(0.95, 0.05, 'Better Reproducibility ↓', transform=ax.transAxes,
            fontsize=10, horizontalalignment='right', style='italic')
    
    # Create custom legend
    activation_patches = [mpatches.Patch(color=color, label=act.replace('_', ' ').upper())
                         for act, color in activation_colors.items()]
    model_patches = [plt.Line2D([0], [0], marker=marker, color='w', 
                               markerfacecolor='gray', markersize=10, label=model.upper())
                    for model, marker in model_markers.items()]
    
    # Only show legends for data we have
    available_activations = set()
    available_models = set()
    for model in multi_model_data:
        available_models.add(model)
        for activation in multi_model_data[model]:
            available_activations.add(activation)
    
    activation_patches = [p for p in activation_patches if any(act in p.get_label().lower() 
                         for act in available_activations)]
    model_patches = [p for p in model_patches if any(mod in p.get_label().lower() 
                    for mod in available_models)]
    
    legend1 = ax.legend(handles=activation_patches, loc='upper left', 
                       title='Activation', fontsize=9, title_fontsize=10)
    ax.add_artist(legend1)
    ax.legend(handles=model_patches, loc='lower right',
             title='Model', fontsize=9, title_fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'📊 Saved accuracy vs reproducibility plot to {save_path}')


def plot_grouped_bar_comparison(multi_model_data: Dict, metric: str = 'relative_pd',
                                save_path: str = 'plots/grouped_bar_comparison.png'):
    """
    Grouped bar chart comparing metric across models and activations.
    
    Args:
        multi_model_data: Hierarchical dict with models and activations
        metric: 'relative_pd', 'val_loss', or 'val_accuracy'
        save_path: Where to save the plot
    """
    models = sorted(multi_model_data.keys())
    all_activations = set()
    for model_data in multi_model_data.values():
        all_activations.update(model_data.keys())
    activations = sorted(all_activations)
    
    # Prepare data
    data_matrix = np.zeros((len(models), len(activations)))
    for i, model in enumerate(models):
        for j, activation in enumerate(activations):
            if activation in multi_model_data[model]:
                data = multi_model_data[model][activation]
                if metric == 'relative_pd':
                    data_matrix[i, j] = data['avg_relative_pd']
                elif metric == 'val_loss':
                    data_matrix[i, j] = data['avg_val_loss']
                elif metric == 'val_accuracy':
                    data_matrix[i, j] = data.get('avg_val_accuracy', 0)
            else:
                data_matrix[i, j] = np.nan
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.15
    multiplier = 0
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(activations)))
    
    for j, activation in enumerate(activations):
        offset = width * multiplier
        values = data_matrix[:, j]
        bars = ax.bar(x + offset, values, width, label=activation.replace('_', ' ').upper(),
                     color=colors[j], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for k, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        multiplier += 1
    
    # Formatting
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    metric_label = metric.replace('_', ' ').title()
    ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_label} Comparison Across Models and Activations',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * (len(activations) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(loc='best', ncol=len(activations), fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'📊 Saved grouped bar comparison to {save_path}')


def plot_training_stability(multi_model_data: Dict,
                           save_path: str = 'plots/training_stability.png'):
    """
    Line plots with confidence bands showing training dynamics across trials.
    Shows mean ± std deviation for each activation function.
    
    Args:
        multi_model_data: Hierarchical dict with models and activations
        save_path: Where to save the plot
    """
    models = sorted(multi_model_data.keys())
    n_models = len(models)
    
    # Create subplots: 2 rows (loss and relative PD), n_models columns
    fig = plt.figure(figsize=(6*n_models, 10))
    gs = GridSpec(2, n_models, figure=fig, hspace=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, model in enumerate(models):
        model_data = multi_model_data[model]
        
        # Top row: Validation loss
        ax_loss = fig.add_subplot(gs[0, idx])
        
        # Bottom row: Would show relative PD over iterations (if available)
        ax_pd = fig.add_subplot(gs[1, idx])
        
        for color_idx, (activation, data) in enumerate(sorted(model_data.items())):
            trials = data.get('trials', [])
            if not trials:
                continue
            
            # Extract loss histories
            loss_histories = []
            for trial in trials:
                history = trial.get('val_loss_history', [])
                if history:
                    loss_histories.append(history)
            
            if loss_histories:
                # Calculate mean and std
                max_len = max(len(h) for h in loss_histories)
                # Pad histories to same length
                padded = []
                for h in loss_histories:
                    if len(h) < max_len:
                        h = list(h) + [h[-1]] * (max_len - len(h))
                    padded.append(h)
                
                histories_array = np.array(padded)
                mean_loss = np.mean(histories_array, axis=0)
                std_loss = np.std(histories_array, axis=0)
                iterations = np.arange(len(mean_loss))
                
                # Plot loss with confidence band
                label = activation.replace('_', ' ').upper()
                ax_loss.plot(iterations, mean_loss, label=label, 
                           color=colors[color_idx], linewidth=2)
                ax_loss.fill_between(iterations, mean_loss - std_loss, mean_loss + std_loss,
                                    color=colors[color_idx], alpha=0.2)
            
            # Plot relative PD (as horizontal line since we only have final value)
            pd_val = data['avg_relative_pd']
            ax_pd.axhline(pd_val, label=activation.replace('_', ' ').upper(),
                         color=colors[color_idx], linewidth=2, linestyle='--')
        
        # Format loss plot
        ax_loss.set_xlabel('Training Iteration', fontsize=10, fontweight='bold')
        if idx == 0:
            ax_loss.set_ylabel('Validation Loss', fontsize=10, fontweight='bold')
        ax_loss.set_title(f'{model.upper()} - Loss ± Std', fontsize=11, fontweight='bold')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc='best', fontsize=8)
        
        # Format PD plot
        ax_pd.set_xlabel('Training Iteration', fontsize=10, fontweight='bold')
        if idx == 0:
            ax_pd.set_ylabel('Relative PD', fontsize=10, fontweight='bold')
        ax_pd.set_title(f'{model.upper()} - Reproducibility', fontsize=11, fontweight='bold')
        ax_pd.grid(True, alpha=0.3)
        ax_pd.legend(loc='best', fontsize=8)
        ax_pd.set_ylim([0.45, 0.55])
    
    fig.suptitle('Training Stability: Mean ± Std Deviation Across Trials',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'📊 Saved training stability plot to {save_path}')


def plot_all_advanced_visualizations(results_dir: str = 'results',
                                     plots_dir: str = 'plots'):
    """
    Generate all advanced visualizations for multi-model comparison.
    
    Args:
        results_dir: Directory containing result JSON files
        plots_dir: Directory to save plots
    """
    # Create plots directory
    Path(plots_dir).mkdir(exist_ok=True)
    
    # Load data
    print('📂 Loading multi-model results...')
    multi_model_data = load_multi_model_results(results_dir)
    
    if not multi_model_data:
        print('⚠️  No results found. Please run experiments first.')
        return
    
    print(f'✓ Loaded data for {len(multi_model_data)} model(s)')
    for model, activations in multi_model_data.items():
        print(f'  - {model.upper()}: {len(activations)} activation(s)')
    
    # Generate all plots
    print('\n🎨 Generating advanced visualizations...')
    
    plot_heatmap_reproducibility(multi_model_data, f'{plots_dir}/heatmap_reproducibility.png')
    
    plot_accuracy_vs_reproducibility(multi_model_data, f'{plots_dir}/accuracy_vs_reproducibility.png')
    
    plot_grouped_bar_comparison(multi_model_data, 'relative_pd', 
                               f'{plots_dir}/grouped_bar_relative_pd.png')
    plot_grouped_bar_comparison(multi_model_data, 'val_loss',
                               f'{plots_dir}/grouped_bar_val_loss.png')
    
    plot_training_stability(multi_model_data, f'{plots_dir}/training_stability.png')
    
    plot_training_evolution(multi_model_data, 'val_loss',
                          f'{plots_dir}/training_evolution_loss.png')
    
    print('\n✅ All advanced visualizations complete!')
    print(f'📁 Saved to {plots_dir}/')


if __name__ == '__main__':
    # Generate all advanced plots
    plot_all_advanced_visualizations()
