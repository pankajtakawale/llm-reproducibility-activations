"""
Plotting utilities for visualizing experiment results.
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import pandas as pd


def plot_training_curves(results, activation_name, save_dir='plots'):
    """
    Plot training and validation loss curves for all trials of an activation.
    
    Args:
        results: Experiment results dictionary
        activation_name: Name of activation function
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot each trial
    for trial in results['trials']:
        trial_id = trial['trial_id']
        train_loss = trial['train_loss_history']
        val_loss = trial['val_loss_history']
        steps = np.arange(len(train_loss)) * 500  # eval_interval
        
        ax1.plot(steps, train_loss, label=f'Trial {trial_id}', alpha=0.7)
        ax2.plot(steps, val_loss, label=f'Trial {trial_id}', alpha=0.7)
    
    # Formatting
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Train Loss')
    ax1.set_title(f'Training Loss - {activation_name.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title(f'Validation Loss - {activation_name.upper()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'{activation_name}_training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved training curves to {save_path}")
    plt.close()


def plot_reproducibility_metrics(results, activation_name, save_dir='plots'):
    """
    Plot reproducibility metrics for an activation function.
    
    Args:
        results: Experiment results dictionary
        activation_name: Name of activation function
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract metrics
    model_pairs = [m['model_pair'] for m in results['reproducibility_metrics']]
    relative_pds = [m['relative_pd'] for m in results['reproducibility_metrics']]
    pred_diffs = [m['prediction_differences'] for m in results['reproducibility_metrics']]
    total_preds = results['reproducibility_metrics'][0]['total_predictions']
    pred_diff_pcts = [d/total_preds*100 for d in pred_diffs]
    
    # Plot relative PD
    ax1.bar(model_pairs, relative_pds, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(relative_pds), color='red', linestyle='--', 
                label=f'Mean: {np.mean(relative_pds):.6f}')
    ax1.set_xlabel('Model Pair')
    ax1.set_ylabel('Relative Prediction Difference')
    ax1.set_title(f'Relative PD - {activation_name.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot prediction differences
    ax2.bar(model_pairs, pred_diff_pcts, color='coral', alpha=0.7)
    ax2.axhline(y=np.mean(pred_diff_pcts), color='red', linestyle='--',
                label=f'Mean: {np.mean(pred_diff_pcts):.2f}%')
    ax2.set_xlabel('Model Pair')
    ax2.set_ylabel('Top-1 Prediction Differences (%)')
    ax2.set_title(f'Prediction Mismatches - {activation_name.upper()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / f'{activation_name}_reproducibility.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved reproducibility metrics to {save_path}")
    plt.close()


def plot_summary_metrics(results, activation_name, save_dir='plots'):
    """
    Plot summary metrics including loss distribution and timing.
    
    Args:
        results: Experiment results dictionary
        activation_name: Name of activation function
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract trial metrics
    val_losses = [t['val_loss'] for t in results['trials']]
    train_times = [t['training_time']/60 for t in results['trials']]  # Convert to minutes
    trial_ids = [f"Trial {t['trial_id']}" for t in results['trials']]
    
    # Plot validation losses
    ax1.bar(trial_ids, val_losses, color='green', alpha=0.6)
    ax1.axhline(y=np.mean(val_losses), color='red', linestyle='--',
                label=f'Mean: {np.mean(val_losses):.4f}')
    ax1.axhline(y=np.mean(val_losses) + np.std(val_losses), color='orange', 
                linestyle=':', alpha=0.7, label=f'Std: ±{np.std(val_losses):.4f}')
    ax1.axhline(y=np.mean(val_losses) - np.std(val_losses), color='orange', 
                linestyle=':', alpha=0.7)
    ax1.set_ylabel('Validation Loss')
    ax1.set_title(f'Validation Loss Distribution - {activation_name.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot training times
    ax2.bar(trial_ids, train_times, color='purple', alpha=0.6)
    ax2.axhline(y=np.mean(train_times), color='red', linestyle='--',
                label=f'Mean: {np.mean(train_times):.2f} min')
    ax2.set_ylabel('Training Time (minutes)')
    ax2.set_title(f'Training Time - {activation_name.upper()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / f'{activation_name}_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved summary metrics to {save_path}")
    plt.close()


def plot_all_activation_results(save_dir='plots'):
    """
    Create comprehensive comparison plots across all activations.
    Reads all result JSON files and generates comparative visualizations.
    
    Args:
        save_dir: Directory containing plots and results
    """
    save_dir = Path(save_dir)
    results_dir = Path('results')
    
    # Load all results
    all_results = {}
    for result_file in results_dir.glob('*_results.json'):
        activation = result_file.stem.replace('_results', '')
        with open(result_file, 'r') as f:
            all_results[activation] = json.load(f)
    
    if not all_results:
        print("⚠️ No results found. Run experiments first.")
        return
    
    print(f"\n📊 Generating comparison plots for {len(all_results)} activations...")
    
    # Prepare data
    activations = list(all_results.keys())
    
    # Create comparison plots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Validation Loss Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    val_losses = [all_results[act]['avg_val_loss'] for act in activations]
    val_stds = [all_results[act]['std_val_loss'] for act in activations]
    x_pos = np.arange(len(activations))
    ax1.bar(x_pos, val_losses, yerr=val_stds, capsize=5, color='steelblue', alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Average Validation Loss (Lower is Better)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Relative PD Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    rel_pds = [all_results[act]['avg_relative_pd'] for act in activations]
    ax2.bar(x_pos, rel_pds, color='coral', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right')
    ax2.set_ylabel('Relative Prediction Difference')
    ax2.set_title('Average Relative PD (Lower is Better)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Training Time Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    train_times = [all_results[act]['avg_training_time']/60 for act in activations]
    ax3.bar(x_pos, train_times, color='green', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right')
    ax3.set_ylabel('Time (minutes)')
    ax3.set_title('Average Training Time')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Reproducibility vs Accuracy Scatter
    ax4 = fig.add_subplot(gs[1, 0])
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(activations)))
    for i, act in enumerate(activations):
        ax4.scatter(all_results[act]['avg_val_loss'], 
                   all_results[act]['avg_relative_pd'],
                   s=200, alpha=0.6, c=[colors_map[i]], label=act.upper())
    ax4.set_xlabel('Validation Loss (Accuracy)')
    ax4.set_ylabel('Relative PD (Reproducibility)')
    ax4.set_title('Reproducibility vs Accuracy Trade-off\n(Bottom-left is ideal)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 5. Loss Variance Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(x_pos, val_stds, color='purple', alpha=0.7)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right')
    ax5.set_ylabel('Standard Deviation')
    ax5.set_title('Validation Loss Variance (Lower is Better)')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Ranking Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate rankings
    rankings = pd.DataFrame({
        'Activation': [a.upper() for a in activations],
        'Val Loss': val_losses,
        'Rel PD': rel_pds,
        'Loss Std': val_stds,
        'Time (min)': train_times
    })
    
    # Normalize and rank (lower is better for all)
    rankings['Repro Score'] = (rankings['Rel PD'] / rankings['Rel PD'].max() + 
                                rankings['Loss Std'] / rankings['Loss Std'].max()) / 2
    rankings['Overall Score'] = (rankings['Val Loss'] / rankings['Val Loss'].max() +
                                  rankings['Repro Score']) / 2
    rankings = rankings.sort_values('Overall Score')
    
    # Display top 3
    summary_text = "🏆 RANKINGS (Overall Score)\n" + "="*30 + "\n\n"
    for i, row in rankings.head(3).iterrows():
        summary_text += f"#{rankings.index.get_loc(i)+1}. {row['Activation']}\n"
        summary_text += f"   Score: {row['Overall Score']:.3f}\n"
        summary_text += f"   Val Loss: {row['Val Loss']:.4f}\n"
        summary_text += f"   Rel PD: {row['Rel PD']:.6f}\n\n"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    
    plt.suptitle('Comprehensive Activation Function Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    save_path = save_dir / 'comprehensive_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comprehensive comparison to {save_path}")
    plt.close()


def save_results_summary(save_path='results/summary.csv'):
    """
    Create a CSV summary of all experiment results.
    
    Args:
        save_path: Path to save CSV file
    """
    results_dir = Path('results')
    
    # Load all results
    all_results = {}
    for result_file in results_dir.glob('*_results.json'):
        activation = result_file.stem.replace('_results', '')
        with open(result_file, 'r') as f:
            all_results[activation] = json.load(f)
    
    if not all_results:
        print("⚠️ No results found.")
        return
    
    # Create summary dataframe
    summary_data = []
    for activation, results in all_results.items():
        # Calculate average prediction differences percentage
        avg_pred_diffs = np.mean([m['prediction_differences'] for m in results['reproducibility_metrics']])
        total_preds = results['reproducibility_metrics'][0]['total_predictions']
        pred_diff_pct = (avg_pred_diffs / total_preds) * 100
        
        summary_data.append({
            'Activation': activation.upper(),
            'Avg_Val_Loss': results['avg_val_loss'],
            'Std_Val_Loss': results['std_val_loss'],
            'Avg_Train_Loss': results['avg_train_loss'],
            'Avg_Relative_PD': results['avg_relative_pd'],
            'Avg_Pred_Diff_Pct': pred_diff_pct,
            'Avg_Training_Time_Sec': results['avg_training_time'],
            'Num_Trials': len(results['trials'])
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Avg_Relative_PD')  # Sort by reproducibility
    
    # Save to CSV
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, float_format='%.6f')
    
    print(f"\n📋 Results Summary:")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\n✅ Saved summary to {save_path}")
    
    return df
