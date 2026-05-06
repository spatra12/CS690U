import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import roc_curve, auc, precision_recall_curve


def visualize_peaks(peaks_df, tf_name, save_prefix=None):
    """
    Create comprehensive visualizations for ChIP-seq peak DataFrame

    Args:
        peaks_df: DataFrame with peak data
        tf_name: Name of transcription factor (for titles)
        save_prefix: Optional prefix for saving files (e.g., 'elf1_chr10')

    Returns:
        Figure object
    """
    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Define grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Color palette
    main_color = '#2E86AB'

    ## ========== PLOT 1: Peak Width Distribution ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(peaks_df['width'], bins=50, edgecolor='black',
             alpha=0.7, color=main_color)
    ax1.axvline(peaks_df['width'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {peaks_df["width"].mean():.0f} bp')
    ax1.axvline(peaks_df['width'].median(), color='green', linestyle='--',
               linewidth=2, label=f'Median: {peaks_df["width"].median():.0f} bp')
    ax1.set_xlabel('Peak Width (bp)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'{tf_name}: Peak Width Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ## ========== PLOT 2: Score Distribution ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(peaks_df['score'], bins=50, edgecolor='black',
             alpha=0.7, color='coral')
    ax2.axvline(peaks_df['score'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {peaks_df["score"].mean():.0f}')
    ax2.set_xlabel('Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'{tf_name}: Peak Score Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ## ========== PLOT 3: Signal Value Distribution ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(peaks_df['signalValue'], bins=50, edgecolor='black',
             alpha=0.7, color='mediumseagreen')
    ax3.axvline(peaks_df['signalValue'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {peaks_df["signalValue"].mean():.1f}')
    ax3.set_xlabel('Signal Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'{tf_name}: Signal Value Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    ## ========== PLOT 4: Score vs Signal Value ==========
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(peaks_df['signalValue'], peaks_df['score'],
                         alpha=0.4, s=20, c=peaks_df['qValue'], cmap='viridis')
    ax4.set_xlabel('Signal Value', fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title(f'{tf_name}: Score vs Signal Value', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('q-Value', fontsize=10)
    ax4.grid(alpha=0.3)

    ## ========== PLOT 5: Peak Positions Along Chromosome ==========
    ax5 = fig.add_subplot(gs[1, 1])
    positions = (peaks_df['start'] + peaks_df['end']) / 2
    ax5.scatter(positions / 1e6, peaks_df['score'], alpha=0.5, s=15, color=main_color)
    ax5.set_xlabel('Position on Chromosome (Mb)', fontsize=11)
    ax5.set_ylabel('Score', fontsize=11)
    ax5.set_title(f'{tf_name}: Peak Distribution Along Chromosome', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)

    ## ========== PLOT 6: q-Value Distribution ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(peaks_df['qValue'], bins=50, edgecolor='black',
             alpha=0.7, color='purple')
    ax6.axvline(peaks_df['qValue'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {peaks_df["qValue"].mean():.1f}')
    ax6.set_xlabel('q-Value (-log10)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title(f'{tf_name}: q-Value Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)

    ## ========== PLOT 7: Quality Categories ==========
    ax7 = fig.add_subplot(gs[2, 0])
    score_bins = [0, 300, 500, 700, 900, 1000]
    score_labels = ['Low\n(0-300)', 'Fair\n(300-500)', 'Good\n(500-700)',
                   'High\n(700-900)', 'Excellent\n(900-1000)']
    peaks_df['score_category'] = pd.cut(peaks_df['score'], bins=score_bins,
                                        labels=score_labels, include_lowest=True)
    score_counts = peaks_df['score_category'].value_counts().sort_index()

    colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
    ax7.bar(range(len(score_counts)), score_counts.values,
           color=colors, edgecolor='black', alpha=0.8)
    ax7.set_xticks(range(len(score_counts)))
    ax7.set_xticklabels(score_counts.index, fontsize=9)
    ax7.set_ylabel('Number of Peaks', fontsize=11)
    ax7.set_title(f'{tf_name}: Peak Quality Categories', fontsize=12, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, v in enumerate(score_counts.values):
        ax7.text(i, v + max(score_counts.values)*0.02, str(v),
                ha='center', fontsize=9, fontweight='bold')

    ## ========== PLOT 8: Width vs Signal Correlation ==========
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(peaks_df['width'], peaks_df['signalValue'],
               alpha=0.4, s=20, color='teal')
    ax8.set_xlabel('Peak Width (bp)', fontsize=11)
    ax8.set_ylabel('Signal Value', fontsize=11)
    ax8.set_title(f'{tf_name}: Width vs Signal Value', fontsize=12, fontweight='bold')

    # Add correlation coefficient
    corr = peaks_df['width'].corr(peaks_df['signalValue'])
    ax8.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax8.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax8.grid(alpha=0.3)

    ## ========== PLOT 9: Summary Statistics ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # Create summary text
    summary_text = f"""
    {tf_name} Peak Statistics
    {'='*35}

    Total Peaks: {len(peaks_df):,}

    Peak Width:
      Mean: {peaks_df['width'].mean():.1f} bp
      Median: {peaks_df['width'].median():.1f} bp
      Range: {peaks_df['width'].min()} - {peaks_df['width'].max():,} bp

    Score:
      Mean: {peaks_df['score'].mean():.1f}
      Median: {peaks_df['score'].median():.1f}

    Signal Value:
      Mean: {peaks_df['signalValue'].mean():.2f}
      Median: {peaks_df['signalValue'].median():.2f}
      Max: {peaks_df['signalValue'].max():.2f}

    q-Value:
      Mean: {peaks_df['qValue'].mean():.2f}
      Median: {peaks_df['qValue'].median():.2f}

    Quality Distribution:
      Excellent (≥900): {len(peaks_df[peaks_df['score'] >= 900]):,}
      High (700-899): {len(peaks_df[(peaks_df['score'] >= 700) & (peaks_df['score'] < 900)]):,}
      Good (500-699): {len(peaks_df[(peaks_df['score'] >= 500) & (peaks_df['score'] < 700)]):,}
      Fair (<500): {len(peaks_df[peaks_df['score'] < 500]):,}
    """

    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Overall title
    fig.suptitle(f'{tf_name} ChIP-seq Peak Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    # Save if prefix provided
    if save_prefix:
        filename = f'{save_prefix}_visualization.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {filename}")

    return fig



def visualize_peaks_simple(peaks_df, tf_name, save_prefix=None):
    """
    Create simple 4-panel visualization

    Args:
        peaks_df: DataFrame with peak data
        tf_name: Name of transcription factor
        save_prefix: Optional prefix for saving

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Peak Width
    axes[0, 0].hist(peaks_df['width'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(peaks_df['width'].mean(), color='red', linestyle='--',
                      label=f'Mean: {peaks_df["width"].mean():.0f}')
    axes[0, 0].set_xlabel('Peak Width (bp)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Peak Width Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Score Distribution
    axes[0, 1].hist(peaks_df['score'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Score Distribution')
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Chromosome Position
    positions = (peaks_df['start'] + peaks_df['end']) / 2
    axes[1, 0].scatter(positions / 1e6, peaks_df['score'], alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Position (Mb)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Peaks Along Chromosome')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Quality Categories
    score_bins = [0, 500, 700, 900, 1000]
    score_labels = ['Fair', 'Good', 'High', 'Excellent']
    peaks_df['quality'] = pd.cut(peaks_df['score'], bins=score_bins, labels=score_labels)
    quality_counts = peaks_df['quality'].value_counts().sort_index()

    axes[1, 1].bar(range(len(quality_counts)), quality_counts.values,
                  color=['#fee08b', '#91cf60', '#66c2a5', '#1a9850'])
    axes[1, 1].set_xticks(range(len(quality_counts)))
    axes[1, 1].set_xticklabels(quality_counts.index)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Peak Quality')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.suptitle(f'{tf_name} Peak Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_prefix:
        filename = f'{save_prefix}_simple.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved to: {filename}")

    return fig


def visualize_sequence_dataset_simple(wt_sequence_df, save_prefix=None):
    """
    Create simple 4-panel visualization

    Args:
        wt_sequence_df: DataFrame with sequence data
        save_prefix: Optional prefix for saving

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Calculate GC content
    def calc_gc(seq):
        if pd.isna(seq) or len(seq) == 0:
            return 0
        return (seq.count('G') + seq.count('C')) / len(seq) * 100

    wt_sequence_df['gc_content'] = wt_sequence_df['seq'].apply(calc_gc)

    # Plot 1: Binding vs Non-binding
    binding_counts = wt_sequence_df['is_binding'].value_counts().sort_index()
    colors = ['#e74c3c', '#2ecc71']
    labels = ['Non-binding', 'Binding']

    axes[0, 0].bar(range(len(binding_counts)), binding_counts.values,
                   color=colors, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xticks(range(len(binding_counts)))
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Binding vs Non-binding Sequences')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add percentages
    for i, count in enumerate(binding_counts.values):
        pct = count / len(wt_sequence_df) * 100
        axes[0, 0].text(i, count, f'{count:,}\n({pct:.1f}%)',
                       ha='center', va='bottom')

    # Plot 2: Sequences per TF
    tf_counts = wt_sequence_df.groupby(['tf_name', 'is_binding']).size().unstack(fill_value=0)

    tf_counts.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#2ecc71'],
                  alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Transcription Factor')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Sequences per TF')
    axes[0, 1].legend(['Non-binding', 'Binding'])
    axes[0, 1].grid(axis='y', alpha=0.3)
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: GC Content Distribution
    binding = wt_sequence_df[wt_sequence_df['is_binding'] == 1]
    non_binding = wt_sequence_df[wt_sequence_df['is_binding'] == 0]

    axes[1, 0].hist(binding['gc_content'], bins=30, alpha=0.6,
                   label='Binding', color='#2ecc71', edgecolor='black')
    axes[1, 0].hist(non_binding['gc_content'], bins=30, alpha=0.6,
                   label='Non-binding', color='#e74c3c', edgecolor='black')
    axes[1, 0].set_xlabel('GC Content (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('GC Content Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Summary text
    axes[1, 1].axis('off')

    total = len(wt_sequence_df)
    bind_count = len(binding)
    non_bind_count = len(non_binding)

    summary = f"""
    Dataset Summary
    {'='*30}

    Total Sequences: {total:,}

    Binding: {bind_count:,} ({bind_count/total*100:.1f}%)
    Non-binding: {non_bind_count:,} ({non_bind_count/total*100:.1f}%)

    Transcription Factors: {wt_sequence_df['tf_name'].nunique()}
    {', '.join(wt_sequence_df['tf_name'].unique())}

    GC Content:
      Binding: {binding['gc_content'].mean():.2f}%
      Non-binding: {non_binding['gc_content'].mean():.2f}%
    """

    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('WT Sequence Dataset Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_prefix:
        filename = f'{save_prefix}_simple.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {filename}")

    return fig


def plot_sequence_lengths(sequences, save_filepath):
    """Plot distribution of sequence lengths."""

    lengths = [len(seq) for seq in sequences]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(lengths), color='red', linestyle='--',
                label=f'Mean: {np.mean(lengths):.0f} bp')
    plt.axvline(np.median(lengths), color='green', linestyle='--',
                label=f'Median: {np.median(lengths):.0f} bp')
    plt.xlabel('Sequence Length (bp)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sequence Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_filepath+'/wildtype_sequence_lengths.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Saved: wildtype_sequence_lengths.png")


#plots for evaluating model performance

def plot_confusion_matrix(y_true, y_pred, save_filepath, title='Confusion Matrix'):
    """Plot confusion matrix heatmap."""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Binding', 'Binding'],
                yticklabels=['No Binding', 'Binding'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)

    # Add accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(1, 2.3, f'Accuracy: {accuracy:.4f}',
             ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{save_filepath}/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved: {save_filepath}/_{title}.png")


def plot_roc_curve(y_true, y_proba, save_filepath, title="ROC Curve"):
    """Plot ROC curve."""

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(f"{save_filepath}/_{title}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved: {save_filepath}/_{title}.png")


def plot_precision_recall_curve(y_true, y_proba, save_filepath, title="Precision-Recall Curve"):
    """Plot Precision-Recall curve."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{save_filepath}/_{title}.png.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved: {save_filepath}/_{title}.png")


#plotting
def create_metrics_table_simple(all_metrics_df, save_path):
    """
    Create simple summary table from metrics

    Args:
        all_metrics_df: DataFrame with columns like 'accuracy', 'f1', 'roc_auc'

    Returns:
        DataFrame with summary statistics
    """
    # Calculate metrics
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'F1-Score', 'ROC AUC'],
        'Mean': [
            float(all_metrics_df['accuracy'].mean()),
            float(all_metrics_df['f1'].mean()),
            float(all_metrics_df['roc_auc'].mean())
        ],
        'Std': [
            float(all_metrics_df['accuracy'].std()),
            float(all_metrics_df['f1'].std()),
            float(all_metrics_df['roc_auc'].std())
        ],
        'Min': [
            float(all_metrics_df['accuracy'].min()),
            float(all_metrics_df['f1'].min()),
            float(all_metrics_df['roc_auc'].min())
        ],
        'Max': [
            float(all_metrics_df['accuracy'].max()),
            float(all_metrics_df['f1'].max()),
            float(all_metrics_df['roc_auc'].max())
        ]
    })

    return metrics_summary

def print_metrics_table(all_metrics_df):
    """
    Print nicely formatted table to console

    Args:
        all_metrics_df: DataFrame with metrics
    """
    # Calculate statistics
    accuracy_mean = float(all_metrics_df['accuracy'].mean())
    accuracy_std = float(all_metrics_df['accuracy'].std())

    f1_mean = float(all_metrics_df['f1'].mean())
    f1_std = float(all_metrics_df['f1'].std())

    roc_auc_mean = float(all_metrics_df['roc_auc'].mean())
    roc_auc_std = float(all_metrics_df['roc_auc'].std())

    # Print formatted table
    print("\n" + "="*70)
    print("VALIDATION METRICS SUMMARY")
    print("="*70)
    print(f"{'Metric':<20} {'Mean':<15} {'Std Dev':<15} {'Range':<20}")
    print("-"*70)
    print(f"{'Accuracy':<20} {accuracy_mean:<15.4f} {accuracy_std:<15.4f} "
          f"{all_metrics_df['accuracy'].min():.4f} - {all_metrics_df['accuracy'].max():.4f}")
    print(f"{'F1-Score':<20} {f1_mean:<15.4f} {f1_std:<15.4f} "
          f"{all_metrics_df['f1'].min():.4f} - {all_metrics_df['f1'].max():.4f}")
    print(f"{'ROC AUC':<20} {roc_auc_mean:<15.4f} {roc_auc_std:<15.4f} "
          f"{all_metrics_df['roc_auc'].min():.4f} - {all_metrics_df['roc_auc'].max():.4f}")
    print("="*70)



def visualize_variants_simple(variant_sequences_df, save_prefix=None):
    """
    Create simple 4-panel visualization

    Args:
        variant_sequences_df: DataFrame with variant sequences
        save_prefix: Optional prefix for saving

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Calculate properties
    variant_sequences_df['alt_allele'] = variant_sequences_df.index.str.split('_').str[-1]
    variant_sequences_df['wt_gc'] = variant_sequences_df['wt'].apply(
        lambda x: (x.count('G') + x.count('C')) / len(x) * 100
    )
    variant_sequences_df['mutant_gc'] = variant_sequences_df['mutant'].apply(
        lambda x: (x.count('G') + x.count('C')) / len(x) * 100
    )

    # Plot 1: Variant counts
    total = len(variant_sequences_df)
    unique = len(set([idx.rsplit('_', 1)[0] for idx in variant_sequences_df.index]))

    axes[0, 0].bar(['Total\nVariants', 'Unique\nPositions'], [total, unique],
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Variant Overview')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Plot 2: Allele distribution
    allele_counts = variant_sequences_df['alt_allele'].value_counts()
    colors = {'A': '#e74c3c', 'T': '#3498db', 'C': '#f39c12', 'G': '#2ecc71'}
    bar_colors = [colors.get(a, '#95a5a6') for a in allele_counts.index]

    axes[0, 1].bar(range(len(allele_counts)), allele_counts.values,
                   color=bar_colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(range(len(allele_counts)))
    axes[0, 1].set_xticklabels(allele_counts.index, fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Alternate Alleles')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Plot 3: GC content scatter
    axes[1, 0].scatter(variant_sequences_df['wt_gc'], variant_sequences_df['mutant_gc'],
                      alpha=0.5, s=50)
    min_gc = min(variant_sequences_df['wt_gc'].min(), variant_sequences_df['mutant_gc'].min())
    max_gc = max(variant_sequences_df['wt_gc'].max(), variant_sequences_df['mutant_gc'].max())
    axes[1, 0].plot([min_gc, max_gc], [min_gc, max_gc], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('WT GC (%)')
    axes[1, 0].set_ylabel('Mutant GC (%)')
    axes[1, 0].set_title('GC Content Comparison')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Summary stats
    axes[1, 1].axis('off')

    summary = f"""
    Summary Statistics
    {'='*25}

    Total Variants: {len(variant_sequences_df)}

    Sequence Length:
      Mean: {variant_sequences_df['wt'].str.len().mean():.0f} bp

    GC Content:
      WT Mean: {variant_sequences_df['wt_gc'].mean():.2f}%
      Mutant Mean: {variant_sequences_df['mutant_gc'].mean():.2f}%

    Alleles:
    """

    for allele, count in allele_counts.items():
        summary += f"\n      {allele}: {count}"

    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Lupus Variant Sequences', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_prefix:
        filename = f'{save_prefix}_simple.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {filename}")

    return fig

#create plot to show the different cosine similarities for different variants

def visualize_cosines_simple(cosines_dict, save_prefix=None):
    """
    Create simple 4-panel visualization

    Args:
        cosines_dict: Dictionary of {variant_id: cosine_similarity}
        save_prefix: Optional prefix for saving

    Returns:
        Figure object
    """
    cosines_df = pd.DataFrame(list(cosines_dict.items()),
                             columns=['variant_id', 'cosine_similarity'])
    cosines_df['impact'] = 1 - cosines_df['cosine_similarity']
    cosines_df['alt_allele'] = cosines_df['variant_id'].str.rsplit('_', n=1).str[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Similarity distribution
    axes[0, 0].hist(cosines_df['cosine_similarity'], bins=30,
                   edgecolor='black', alpha=0.7, color='#3498db')
    axes[0, 0].axvline(cosines_df['cosine_similarity'].mean(),
                      color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Similarity Distribution')
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Top 10 impacts
    top10 = cosines_df.nlargest(10, 'impact')
    axes[0, 1].barh(range(len(top10)), top10['impact'], color='#e74c3c', alpha=0.7)
    axes[0, 1].set_yticks(range(len(top10)))
    axes[0, 1].set_yticklabels(top10['variant_id'], fontsize=8)
    axes[0, 1].set_xlabel('Impact')
    axes[0, 1].set_title('Top 10 Highest Impact')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Plot 3: Impact by allele
    mean_impact = cosines_df.groupby('alt_allele')['impact'].mean().sort_values(ascending=False)
    colors = {'A': '#e74c3c', 'T': '#3498db', 'C': '#f39c12', 'G': '#2ecc71'}
    bar_colors = [colors.get(a, '#95a5a6') for a in mean_impact.index]

    axes[1, 0].bar(range(len(mean_impact)), mean_impact.values,
                  color=bar_colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(len(mean_impact)))
    axes[1, 0].set_xticklabels(mean_impact.index, fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Mean Impact')
    axes[1, 0].set_title('Impact by Allele')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Plot 4: Summary
    axes[1, 1].axis('off')

    summary = f"""
    Summary Statistics
    {'='*30}

    Total Variants: {len(cosines_df)}

    Cosine Similarity:
      Mean: {cosines_df['cosine_similarity'].mean():.4f}
      Std:  {cosines_df['cosine_similarity'].std():.4f}

    Impact (1 - Similarity):
      Mean: {cosines_df['impact'].mean():.4f}
      Max:  {cosines_df['impact'].max():.4f}

    Top Impact Variant:
      {top10.iloc[0]['variant_id']}
      Impact: {top10.iloc[0]['impact']:.4f}
    """

    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Cosine Similarity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_prefix:
        filename = f'{save_prefix}_simple.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {filename}")

    return fig


def visualize_risk_simple(risk_scores_dict, save_prefix=None):
    """
    Create simple 4-panel visualization

    Args:
        risk_scores_dict: Dictionary of {variant_id: delta}
        save_prefix: Optional prefix for saving

    Returns:
        Figure object
    """
    risk_df = pd.DataFrame(list(risk_scores_dict.items()),
                          columns=['variant_id', 'delta'])
    risk_df['alt_allele'] = risk_df['variant_id'].str.rsplit('_', n=1).str[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Distribution
    axes[0, 0].hist(risk_df['delta'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Δ Risk Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Risk Score Distribution')
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Top 10 Loss
    top10_loss = risk_df.nlargest(10, 'delta')
    axes[0, 1].barh(range(len(top10_loss)), top10_loss['delta'],
                   color='#e74c3c', alpha=0.7)
    axes[0, 1].set_yticks(range(len(top10_loss)))
    axes[0, 1].set_yticklabels(top10_loss['variant_id'], fontsize=8)
    axes[0, 1].set_xlabel('Δ Risk')
    axes[0, 1].set_title('Top 10: Loss of Binding')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Plot 3: Risk by allele
    mean_risk = risk_df.groupby('alt_allele')['delta'].mean().sort_values(ascending=False)
    colors = {'A': '#e74c3c', 'T': '#3498db', 'C': '#f39c12', 'G': '#2ecc71'}
    bar_colors = [colors.get(a, '#95a5a6') for a in mean_risk.index]

    axes[1, 0].bar(range(len(mean_risk)), mean_risk.values,
                  color=bar_colors, alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xticks(range(len(mean_risk)))
    axes[1, 0].set_xticklabels(mean_risk.index, fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Mean Δ Risk')
    axes[1, 0].set_title('Risk by Allele')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Plot 4: Summary
    axes[1, 1].axis('off')

    n_loss = len(risk_df[risk_df['delta'] > 0.05])
    n_gain = len(risk_df[risk_df['delta'] < -0.05])
    n_neutral = len(risk_df[abs(risk_df['delta']) <= 0.05])

    summary = f"""
    Summary Statistics
    {'='*30}

    Total Variants: {len(risk_df)}

    Risk Score:
      Mean: {risk_df['delta'].mean():.4f}
      Std:  {risk_df['delta'].std():.4f}

    Categories:
      Loss (>0.05): {n_loss}
      Neutral: {n_neutral}
      Gain (<-0.05): {n_gain}

    Highest Risk:
      {top10_loss.iloc[0]['variant_id']}
      Δ = {top10_loss.iloc[0]['delta']:.4f}
    """

    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.suptitle('Risk Score Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_prefix:
        filename = f'{save_prefix}_simple.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved to: {filename}")

    return fig

def visualize_pwm_simple(pwm_scores_df, save_prefix=None):
    """
    Create simple 4-panel visualization

    Args:
        pwm_scores_df: DataFrame with PWM scores
        save_prefix: Optional prefix for saving

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    pwm_scores_df['alt_allele'] = pwm_scores_df.index.str.rsplit('_', n=1).str[-1]

    # Plot 1: STAT5A/B distribution
    axes[0, 0].hist(pwm_scores_df['delta_pwm_stat5ab'], bins=30,
                   edgecolor='black', alpha=0.7, color='#3498db')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Δ PWM Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('STAT5A/B: PWM Score Changes')
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: ELF1 distribution
    axes[0, 1].hist(pwm_scores_df['delta_pwm_elf1'], bins=30,
                   edgecolor='black', alpha=0.7, color='#e74c3c')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Δ PWM Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('ELF1: PWM Score Changes')
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Correlation
    axes[1, 0].scatter(pwm_scores_df['delta_pwm_stat5ab'],
                      pwm_scores_df['delta_pwm_elf1'],
                      alpha=0.5, s=50)
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Δ PWM STAT5A/B')
    axes[1, 0].set_ylabel('Δ PWM ELF1')
    axes[1, 0].set_title('STAT5A/B vs ELF1')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Summary
    axes[1, 1].axis('off')

    summary = f"""
    PWM Score Summary
    {'='*30}

    Total Variants: {len(pwm_scores_df)}

    STAT5A/B:
      Mean Δ: {pwm_scores_df['delta_pwm_stat5ab'].mean():.4f}
      Loss: {len(pwm_scores_df[pwm_scores_df['delta_pwm_stat5ab'] < 0])}

    ELF1:
      Mean Δ: {pwm_scores_df['delta_pwm_elf1'].mean():.4f}
      Loss: {len(pwm_scores_df[pwm_scores_df['delta_pwm_elf1'] < 0])}

    Correlation: {pwm_scores_df['delta_pwm_stat5ab'].corr(pwm_scores_df['delta_pwm_elf1']):.3f}
    """

    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

    plt.suptitle('PWM Score Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_prefix:
        filename = f'{save_prefix}_simple.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {filename}")

    return fig