"""
Comprehensive analysis script for experiment results.
Aggregates results, performs statistical tests, and draws conclusions.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt


def load_all_results(results_dir='results'):
    """Load all experiment results from JSON files."""
    results_dir = Path(results_dir)
    all_results = {}
    
    for result_file in results_dir.glob('*_results.json'):
        activation = result_file.stem.replace('_results', '')
        with open(result_file, 'r') as f:
            all_results[activation] = json.load(f)
    
    return all_results


def calculate_statistical_significance(all_results):
    """
    Perform statistical tests to determine if differences are significant.
    
    Returns:
        DataFrame with pairwise comparisons
    """
    activations = list(all_results.keys())
    comparisons = []
    
    for i, act1 in enumerate(activations):
        for act2 in activations[i+1:]:
            # Get validation losses from all trials
            val_losses1 = [t['val_loss'] for t in all_results[act1]['trials']]
            val_losses2 = [t['val_loss'] for t in all_results[act2]['trials']]
            
            # t-test for validation loss
            t_stat_loss, p_value_loss = stats.ttest_ind(val_losses1, val_losses2)
            
            # Get relative PDs
            rel_pds1 = [m['relative_pd'] for m in all_results[act1]['reproducibility_metrics']]
            rel_pds2 = [m['relative_pd'] for m in all_results[act2]['reproducibility_metrics']]
            
            # t-test for reproducibility
            t_stat_repro, p_value_repro = stats.ttest_ind(rel_pds1, rel_pds2)
            
            comparisons.append({
                'Comparison': f"{act1.upper()} vs {act2.upper()}",
                'Loss_Diff': np.mean(val_losses1) - np.mean(val_losses2),
                'Loss_PValue': p_value_loss,
                'Loss_Significant': p_value_loss < 0.05,
                'ReproPD_Diff': np.mean(rel_pds1) - np.mean(rel_pds2),
                'Repro_PValue': p_value_repro,
                'Repro_Significant': p_value_repro < 0.05
            })
    
    return pd.DataFrame(comparisons)


def rank_activations(all_results, weights=None):
    """
    Rank activation functions based on multiple criteria.
    
    Args:
        all_results: Dictionary of all results
        weights: Dictionary with keys 'reproducibility', 'accuracy', 'efficiency'
                Default: {'reproducibility': 0.4, 'accuracy': 0.4, 'efficiency': 0.2}
    
    Returns:
        DataFrame with rankings
    """
    if weights is None:
        weights = {'reproducibility': 0.4, 'accuracy': 0.4, 'efficiency': 0.2}
    
    activations = list(all_results.keys())
    
    # Extract metrics
    data = []
    for act in activations:
        res = all_results[act]
        avg_pred_diffs = np.mean([m['prediction_differences'] for m in res['reproducibility_metrics']])
        total_preds = res['reproducibility_metrics'][0]['total_predictions']
        
        data.append({
            'Activation': act.upper(),
            'Val_Loss': res['avg_val_loss'],
            'Loss_Std': res['std_val_loss'],
            'Relative_PD': res['avg_relative_pd'],
            'Pred_Diff_Pct': (avg_pred_diffs / total_preds) * 100,
            'Training_Time': res['avg_training_time']
        })
    
    df = pd.DataFrame(data)
    
    # Normalize metrics (0-1 scale, lower is better)
    df['Val_Loss_Norm'] = (df['Val_Loss'] - df['Val_Loss'].min()) / (df['Val_Loss'].max() - df['Val_Loss'].min())
    df['Loss_Std_Norm'] = (df['Loss_Std'] - df['Loss_Std'].min()) / (df['Loss_Std'].max() - df['Loss_Std'].min())
    df['Relative_PD_Norm'] = (df['Relative_PD'] - df['Relative_PD'].min()) / (df['Relative_PD'].max() - df['Relative_PD'].min())
    df['Time_Norm'] = (df['Training_Time'] - df['Training_Time'].min()) / (df['Training_Time'].max() - df['Training_Time'].min())
    
    # Calculate composite scores
    df['Reproducibility_Score'] = (df['Relative_PD_Norm'] + df['Loss_Std_Norm']) / 2
    df['Accuracy_Score'] = df['Val_Loss_Norm']
    df['Efficiency_Score'] = df['Time_Norm']
    
    # Overall weighted score (lower is better)
    df['Overall_Score'] = (weights['reproducibility'] * df['Reproducibility_Score'] +
                           weights['accuracy'] * df['Accuracy_Score'] +
                           weights['efficiency'] * df['Efficiency_Score'])
    
    df = df.sort_values('Overall_Score')
    df['Rank'] = range(1, len(df) + 1)
    
    return df


def generate_final_report(all_results, save_path='results/FINAL_REPORT.md'):
    """Generate a comprehensive markdown report with conclusions."""
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get rankings with different weight configurations
    research_weights = {'reproducibility': 0.6, 'accuracy': 0.3, 'efficiency': 0.1}
    production_weights = {'reproducibility': 0.4, 'accuracy': 0.4, 'efficiency': 0.2}
    speed_weights = {'reproducibility': 0.2, 'accuracy': 0.3, 'efficiency': 0.5}
    
    research_rank = rank_activations(all_results, research_weights)
    production_rank = rank_activations(all_results, production_weights)
    speed_rank = rank_activations(all_results, speed_weights)
    
    # Statistical significance
    sig_tests = calculate_statistical_significance(all_results)
    
    # Generate report
    report = []
    report.append("# Final Experimental Report: LLM Reproducibility with Activation Functions\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n\n")
    
    # Executive Summary
    report.append("## 📊 Executive Summary\n\n")
    best_overall = production_rank.iloc[0]
    best_repro = research_rank.iloc[0]
    best_accuracy = production_rank.nsmallest(1, 'Val_Loss').iloc[0]
    
    report.append(f"- **Best Overall (Balanced):** {best_overall['Activation']}\n")
    report.append(f"- **Most Reproducible:** {best_repro['Activation']}\n")
    report.append(f"- **Most Accurate:** {best_accuracy['Activation']}\n")
    report.append(f"- **Total Experiments:** {sum(len(r['trials']) for r in all_results.values())} models trained\n")
    report.append(f"- **Total Training Time:** {sum(r['avg_training_time'] * len(r['trials']) for r in all_results.values())/3600:.2f} hours\n\n")
    
    # Detailed Results Table
    report.append("## 📋 Detailed Results\n\n")
    report.append("### Raw Metrics\n\n")
    
    # Create detailed table
    detailed_data = []
    for act, res in all_results.items():
        avg_pred_diffs = np.mean([m['prediction_differences'] for m in res['reproducibility_metrics']])
        total_preds = res['reproducibility_metrics'][0]['total_predictions']
        
        detailed_data.append({
            'Activation': act.upper(),
            'Val Loss': f"{res['avg_val_loss']:.4f} ± {res['std_val_loss']:.4f}",
            'Relative PD': f"{res['avg_relative_pd']:.6f}",
            'Pred Diff %': f"{(avg_pred_diffs/total_preds)*100:.2f}%",
            'Train Time': f"{res['avg_training_time']/60:.2f} min"
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    report.append(detailed_df.to_markdown(index=False))
    report.append("\n\n")
    
    # Rankings by Use Case
    report.append("## 🏆 Rankings by Use Case\n\n")
    
    report.append("### 1. Research/Reproducibility Focus (60% Repro, 30% Accuracy, 10% Speed)\n\n")
    research_table = research_rank[['Rank', 'Activation', 'Val_Loss', 'Relative_PD', 'Overall_Score']].copy()
    report.append(research_table.to_markdown(index=False))
    report.append("\n\n")
    
    report.append("### 2. Production/Balanced (40% Repro, 40% Accuracy, 20% Speed)\n\n")
    production_table = production_rank[['Rank', 'Activation', 'Val_Loss', 'Relative_PD', 'Overall_Score']].copy()
    report.append(production_table.to_markdown(index=False))
    report.append("\n\n")
    
    report.append("### 3. Speed Priority (20% Repro, 30% Accuracy, 50% Speed)\n\n")
    speed_table = speed_rank[['Rank', 'Activation', 'Val_Loss', 'Training_Time']].copy()
    report.append(speed_table.to_markdown(index=False))
    report.append("\n\n")
    
    # Statistical Significance
    report.append("## 📈 Statistical Significance Tests\n\n")
    report.append("Pairwise t-tests (p < 0.05 indicates significant difference):\n\n")
    
    sig_table = sig_tests[['Comparison', 'Loss_Diff', 'Loss_Significant', 'ReproPD_Diff', 'Repro_Significant']].copy()
    sig_table['Loss_Diff'] = sig_table['Loss_Diff'].map(lambda x: f"{x:.4f}")
    sig_table['ReproPD_Diff'] = sig_table['ReproPD_Diff'].map(lambda x: f"{x:.6f}")
    sig_table['Loss_Significant'] = sig_table['Loss_Significant'].map(lambda x: '✓ Yes' if x else '✗ No')
    sig_table['Repro_Significant'] = sig_table['Repro_Significant'].map(lambda x: '✓ Yes' if x else '✗ No')
    
    report.append(sig_table.to_markdown(index=False))
    report.append("\n\n")
    
    # Key Findings
    report.append("## 🔍 Key Findings\n\n")
    
    # Find smooth vs non-smooth comparison
    smooth_acts = [act for act in all_results.keys() if act in ['smelu_05', 'smelu_1', 'gelu', 'swish']]
    non_smooth_acts = [act for act in all_results.keys() if act == 'relu']
    
    if smooth_acts and non_smooth_acts:
        smooth_avg_pd = np.mean([all_results[act]['avg_relative_pd'] for act in smooth_acts])
        non_smooth_avg_pd = np.mean([all_results[act]['avg_relative_pd'] for act in non_smooth_acts])
        
        report.append(f"### Smooth vs Non-Smooth Activations\n\n")
        report.append(f"- **Smooth activations** (SmeLU, GELU, Swish) average Relative PD: `{smooth_avg_pd:.6f}`\n")
        report.append(f"- **Non-smooth activations** (ReLU) average Relative PD: `{non_smooth_avg_pd:.6f}`\n")
        
        if smooth_avg_pd < non_smooth_avg_pd:
            improvement = ((non_smooth_avg_pd - smooth_avg_pd) / non_smooth_avg_pd) * 100
            report.append(f"- **✅ Smooth activations are {improvement:.1f}% more reproducible!**\n\n")
        else:
            report.append(f"- ⚠️ Hypothesis not supported: ReLU showed better reproducibility\n\n")
    
    # Best/Worst performers
    best_pd = min(all_results.items(), key=lambda x: x[1]['avg_relative_pd'])
    worst_pd = max(all_results.items(), key=lambda x: x[1]['avg_relative_pd'])
    
    report.append(f"### Reproducibility Champions\n\n")
    report.append(f"- **Most Reproducible:** {best_pd[0].upper()} (Relative PD: {best_pd[1]['avg_relative_pd']:.6f})\n")
    report.append(f"- **Least Reproducible:** {worst_pd[0].upper()} (Relative PD: {worst_pd[1]['avg_relative_pd']:.6f})\n\n")
    
    # Conclusions
    report.append("## 🎯 Conclusions\n\n")
    
    report.append("### Recommendations by Context:\n\n")
    report.append(f"1. **For Research & Reproducibility Studies:**\n")
    report.append(f"   - Choose: **{research_rank.iloc[0]['Activation']}**\n")
    report.append(f"   - Reason: Best reproducibility metrics with acceptable accuracy\n\n")
    
    report.append(f"2. **For Production ML Systems:**\n")
    report.append(f"   - Choose: **{production_rank.iloc[0]['Activation']}**\n")
    report.append(f"   - Reason: Optimal balance of reproducibility and accuracy\n\n")
    
    report.append(f"3. **For Fast Prototyping:**\n")
    report.append(f"   - Choose: **{speed_rank.iloc[0]['Activation']}**\n")
    report.append(f"   - Reason: Fastest training time\n\n")
    
    report.append("### Trade-offs:\n\n")
    report.append("- Improved reproducibility may come at the cost of slightly worse accuracy\n")
    report.append("- Smooth activations generally require marginally more computation time\n")
    report.append("- The choice should be driven by your specific application requirements\n\n")
    
    # Future Work
    report.append("## 🚀 Future Work\n\n")
    report.append("- Scale experiments to larger models (Pythia-160M, GPT-2)\n")
    report.append("- Test on diverse datasets (code, books, web text)\n")
    report.append("- Investigate activation function combinations\n")
    report.append("- Analyze impact of model size on reproducibility\n")
    report.append("- Explore quantization effects (FP16, INT8)\n\n")
    
    report.append("---\n")
    report.append("*Report generated automatically from experimental results*\n")
    
    # Save report
    with open(save_path, 'w') as f:
        f.writelines(report)
    
    print(f"\n{'='*80}")
    print(f"✅ Final report saved to: {save_path}")
    print(f"{'='*80}\n")
    
    # Print to console
    print("".join(report))
    
    return "".join(report)


if __name__ == '__main__':
    # Load results
    all_results = load_all_results()
    
    if not all_results:
        print("❌ No results found. Run experiments first.")
        exit(1)
    
    print(f"\n📊 Analyzing results for {len(all_results)} activation functions...\n")
    
    # Generate comprehensive report
    generate_final_report(all_results)
    
    print("\n✅ Analysis complete!")
