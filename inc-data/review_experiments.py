
#!/usr/bin/env python3
"""
Experiment Review Script
Loads and reviews all experiment data from inc_vs_batch_v4/ directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='Review experiment results')
parser.add_argument('--paper', action='store_true', help='Generate paper-style plots in plots-paper/')
args = parser.parse_args()

# Configuration
DATA_DIR = Path("inc_vs_batch_v4")

if args.paper:
    OUTPUT_DIR = Path("plots-paper")
    # Paper styling from plots.ipynb
    MARKER_SIZE = 8
    FONT_SIZE = 14
    FIGSIZE = (4, 3)
    plt.rc('font', size=FONT_SIZE)
    plt.rc('axes', labelsize=FONT_SIZE, titlesize=FONT_SIZE)
    plt.rc('xtick', labelsize=FONT_SIZE)
    plt.rc('ytick', labelsize=FONT_SIZE)
    plt.rc('legend', fontsize=FONT_SIZE)
    # Don't use seaborn style for paper
else:
    OUTPUT_DIR = Path("review_plots")
    # Review styling
    plt.style.use('seaborn-v0_8-whitegrid')
    FIGSIZE = (10, 6)

OUTPUT_DIR.mkdir(exist_ok=True)

# Style map for paper plots
STYLE_MAP = {
    'IncMiner':   {'color': 'C0', 'marker': 'o', 'markersize': MARKER_SIZE if args.paper else 8},
    'BatchMiner': {'color': 'C1', 'marker': 's', 'markersize': MARKER_SIZE if args.paper else 8},
}


def load_all_csvs(data_dir):
    """Load all CSV files from the data directory."""
    csv_files = sorted(glob.glob(str(data_dir / "experiment_results_*.csv")))

    datasets = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        dataset_name = filename.replace("experiment_results_", "").replace(".csv", "")

        # Skip AMiner and dblp_ml datasets
        if dataset_name.startswith('AMiner_') or dataset_name == 'dblp_ml':
            continue

        try:
            df = pd.read_csv(csv_file)
            datasets[dataset_name] = df
            print(f"✓ Loaded {dataset_name}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Failed to load {dataset_name}: {e}")

    return datasets


def summarize_dataset(name, df):
    """Print summary statistics for a dataset."""
    print(f"\n{'='*80}")
    print(f"Dataset: {name}")
    print(f"{'='*80}")

    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if 'experiment_type' in df.columns:
        exp_types = df['experiment_type'].value_counts()
        print(f"\nExperiment types:\n{exp_types.to_string()}")

    if 'inc_type' in df.columns:
        inc_types = df['inc_type'].value_counts()
        print(f"\nIncrement types:\n{inc_types.to_string()}")

    # Find varying parameters
    print(f"\nVarying parameters:")
    for col in df.columns:
        if df[col].nunique() > 1 and df[col].nunique() < len(df):
            print(f"  - {col}: {df[col].nunique()} unique values")
            if df[col].nunique() <= 10:
                print(f"    Values: {sorted(df[col].unique())}")

    # Key metrics
    print(f"\nKey metrics:")
    if 'inc_time_s' in df.columns:
        print(f"  - inc_time_s: min={df['inc_time_s'].min():.3f}s, max={df['inc_time_s'].max():.3f}s, mean={df['inc_time_s'].mean():.3f}s")
    if 'batch_time_s' in df.columns:
        print(f"  - batch_time_s: min={df['batch_time_s'].min():.3f}s, max={df['batch_time_s'].max():.3f}s, mean={df['batch_time_s'].mean():.3f}s")
    if 'inc_recall' in df.columns:
        print(f"  - inc_recall: min={df['inc_recall'].min():.4f}, max={df['inc_recall'].max():.4f}, mean={df['inc_recall'].mean():.4f}")

    # Speedup statistics
    if 'inc_time_s' in df.columns and 'batch_time_s' in df.columns:
        print(f"\n{'='*80}")
        print(f"Speedup Analysis (IncMiner over BatchMiner)")
        print(f"{'='*80}")

        # Calculate speedup where both times are available and > 0
        valid_df = df[(df['inc_time_s'] > 0) & (df['batch_time_s'] > 0)].copy()
        if len(valid_df) > 0:
            valid_df['speedup'] = valid_df['batch_time_s'] / valid_df['inc_time_s']

            print(f"\nOverall (all experiments with valid times):")
            print(f"  - Speedup: min={valid_df['speedup'].min():.2f}x, max={valid_df['speedup'].max():.2f}x, mean={valid_df['speedup'].mean():.2f}x, median={valid_df['speedup'].median():.2f}x")

            # Break down by experiment type
            if 'experiment_type' in valid_df.columns:
                for exp_type in sorted(valid_df['experiment_type'].unique()):
                    exp_df = valid_df[valid_df['experiment_type'] == exp_type]
                    if len(exp_df) > 1:
                        print(f"\n  {exp_type}:")
                        print(f"    Speedup: min={exp_df['speedup'].min():.2f}x, max={exp_df['speedup'].max():.2f}x, mean={exp_df['speedup'].mean():.2f}x")
                        # Show per-data-point speedups if it's an ADD/DELETE experiment with inc_ratio
                        if 'inc_ratio' in exp_df.columns and exp_type in ['ADD', 'DELETE']:
                            print(f"    Per increment ratio:")
                            for _, row in exp_df.sort_values('inc_ratio').iterrows():
                                print(f"      inc_ratio={row['inc_ratio']:.2f}: {row['speedup']:.2f}x (Inc={row['inc_time_s']:.2f}s, Batch={row['batch_time_s']:.2f}s)")


def plot_experiment(name, df, output_dir):
    """Generate plots for a dataset."""
    plots_generated = []

    if 'experiment_type' not in df.columns:
        return plots_generated

    # Helper to calculate speedup
    def get_speedup_stats(sub_df):
        if 'inc_time_s' in sub_df.columns and 'batch_time_s' in sub_df.columns:
            valid = sub_df[(sub_df['inc_time_s'] > 0) & (sub_df['batch_time_s'] > 0)]
            if len(valid) > 0:
                speedup = valid['batch_time_s'] / valid['inc_time_s']
                return f" (speedup: {speedup.min():.2f}x-{speedup.max():.2f}x, mean={speedup.mean():.2f}x)"
        return ""

    # Get plot styles based on mode
    def get_style(label):
        if args.paper and label in STYLE_MAP:
            return STYLE_MAP[label]
        return {}

    # Filter for ADD experiments if available
    if 'inc_type' in df.columns:
        add_df = df[(df['experiment_type'] == 'ADD') & (df['inc_type'] == 'add')]
        if len(add_df) > 0:
            speedup_str = get_speedup_stats(add_df) if not args.paper else ""
            # Plot: Time vs inc_ratio
            if 'inc_ratio' in add_df.columns and 'inc_time_s' in add_df.columns:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                plot_df = add_df.sort_values('inc_ratio')
                ax.plot(plot_df['inc_ratio'], plot_df['inc_time_s'], label='IncMiner', linewidth=2, **get_style('IncMiner'))
                if 'batch_time_s' in plot_df.columns:
                    ax.plot(plot_df['inc_ratio'], plot_df['batch_time_s'], label='BatchMiner', linewidth=2, **get_style('BatchMiner'))
                if not args.paper:
                    ax.set_xlabel('Increment ratio')
                ax.set_ylabel('Time (s)')
                if args.paper:
                    pass  # No title in paper mode
                else:
                    ax.set_title(f'{name} - Time vs Increment Ratio (ADD){speedup_str}')
                if not args.paper:
                    ax.legend(frameon=True)
                if not args.paper:
                    ax.grid(True, alpha=0.3)
                output_path = output_dir / f"{name}_add_time.pdf"
                fig.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
                plots_generated.append(output_path)

            # Plot: Recall vs inc_ratio
            if 'inc_ratio' in add_df.columns and 'inc_recall' in add_df.columns:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                plot_df = add_df.sort_values('inc_ratio')
                recall_style = {}
                if not args.paper:
                    recall_style = {'marker': 'o', 'color': 'green'}
                ax.plot(plot_df['inc_ratio'], plot_df['inc_recall'], linewidth=2, **recall_style)
                if not args.paper:
                    ax.set_xlabel('Increment ratio')
                ax.set_ylabel('Recall')
                if not args.paper:
                    ax.set_title(f'{name} - Recall vs Increment Ratio (ADD)')
                ax.set_ylim([0, 1.05])
                if not args.paper:
                    ax.grid(True, alpha=0.3)
                output_path = output_dir / f"{name}_add_recall.pdf"
                fig.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
                plots_generated.append(output_path)

        # DELETE experiments
        del_df = df[(df['experiment_type'] == 'DELETE') & (df['inc_type'] == 'delete')]
        if len(del_df) > 0:
            speedup_str = get_speedup_stats(del_df) if not args.paper else ""
            # Plot: Time vs inc_ratio
            if 'inc_ratio' in del_df.columns and 'inc_time_s' in del_df.columns:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                plot_df = del_df.sort_values('inc_ratio')
                inc_style = {}
                batch_style = {}
                if not args.paper:
                    inc_style = {'marker': 'o', 'color': 'red'}
                    batch_style = {'marker': 's', 'color': 'orange'}
                ax.plot(plot_df['inc_ratio'], plot_df['inc_time_s'], label='IncMiner', linewidth=2, **get_style('IncMiner'), **inc_style)
                if 'batch_time_s' in plot_df.columns:
                    ax.plot(plot_df['inc_ratio'], plot_df['batch_time_s'], label='BatchMiner', linewidth=2, **get_style('BatchMiner'), **batch_style)
                if not args.paper:
                    ax.set_xlabel('Increment ratio')
                ax.set_ylabel('Time (s)')
                if args.paper:
                    pass  # No title in paper mode
                else:
                    ax.set_title(f'{name} - Time vs Increment Ratio (DELETE){speedup_str}')
                if not args.paper:
                    ax.legend(frameon=True)
                if not args.paper:
                    ax.grid(True, alpha=0.3)
                output_path = output_dir / f"{name}_delete_time.pdf"
                fig.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
                plots_generated.append(output_path)

    # SCALE experiments
    scale_df = df[df['experiment_type'] == 'SCALE']
    if len(scale_df) > 0:
        speedup_str = get_speedup_stats(scale_df) if not args.paper else ""
        if 'scale' in scale_df.columns and 'inc_time_s' in scale_df.columns:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_df = scale_df.sort_values('scale')
            inc_style = {}
            batch_style = {}
            if not args.paper:
                inc_style = {'marker': 'o'}
                batch_style = {'marker': 's'}
            ax.plot(plot_df['scale'], plot_df['inc_time_s'], label='IncMiner', linewidth=2, **get_style('IncMiner'), **inc_style)
            if 'batch_time_s' in plot_df.columns:
                ax.plot(plot_df['scale'], plot_df['batch_time_s'], label='BatchMiner', linewidth=2, **get_style('BatchMiner'), **batch_style)
            if not args.paper:
                ax.set_xlabel('Scale factor')
            ax.set_ylabel('Time (s)')
            if args.paper:
                pass  # No title in paper mode
            else:
                ax.set_title(f'{name} - Time vs Scale (SCALE){speedup_str}')
            if not args.paper:
                ax.legend(frameon=True)
            if not args.paper:
                ax.grid(True, alpha=0.3)
            output_path = output_dir / f"{name}_scale_time.pdf"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            plots_generated.append(output_path)

    # SUPPORT experiments
    support_df = df[df['experiment_type'] == 'SUPPORT']
    if len(support_df) > 0:
        speedup_str = get_speedup_stats(support_df) if not args.paper else ""
        if 'support' in support_df.columns and 'inc_time_s' in support_df.columns:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_df = support_df.sort_values('support')
            inc_style = {}
            batch_style = {}
            if not args.paper:
                inc_style = {'marker': 'o', 'color': 'purple'}
                batch_style = {'marker': 's', 'color': 'brown'}
            ax.plot(plot_df['support'], plot_df['inc_time_s'], label='IncMiner', linewidth=2, **get_style('IncMiner'), **inc_style)
            if 'batch_time_s' in plot_df.columns:
                ax.plot(plot_df['support'], plot_df['batch_time_s'], label='BatchMiner', linewidth=2, **get_style('BatchMiner'), **batch_style)
            if not args.paper:
                ax.set_xlabel('Support')
            ax.set_xscale('log')
            ax.set_ylabel('Time (s)')
            if args.paper:
                pass  # No title in paper mode
            else:
                ax.set_title(f'{name} - Time vs Support{speedup_str}')
            if not args.paper:
                ax.legend(frameon=True)
            if not args.paper:
                ax.grid(True, alpha=0.3)
            output_path = output_dir / f"{name}_support_time.pdf"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            plots_generated.append(output_path)

        # Plot: Recall vs support
        if 'support' in support_df.columns and 'inc_recall' in support_df.columns:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_df = support_df.sort_values('support')
            recall_style = {}
            if not args.paper:
                recall_style = {'marker': 'o', 'color': 'purple'}
            ax.plot(plot_df['support'], plot_df['inc_recall'], linewidth=2, **recall_style)
            if not args.paper:
                ax.set_xlabel('Support')
            ax.set_xscale('log')
            ax.set_ylabel('Recall')
            if not args.paper:
                ax.set_title(f'{name} - Recall vs Support')
            ax.set_ylim([0, 1.05])
            if not args.paper:
                ax.grid(True, alpha=0.3)
            output_path = output_dir / f"{name}_support_recall.pdf"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            plots_generated.append(output_path)

    # CONFIDENCE experiments
    conf_df = df[df['experiment_type'] == 'CONFIDENCE']
    if len(conf_df) > 0:
        speedup_str = get_speedup_stats(conf_df) if not args.paper else ""
        if 'confidence' in conf_df.columns and 'inc_time_s' in conf_df.columns:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_df = conf_df.sort_values('confidence')
            inc_style = {}
            batch_style = {}
            if not args.paper:
                inc_style = {'marker': 'o', 'color': 'cyan'}
                batch_style = {'marker': 's', 'color': 'teal'}
            ax.plot(plot_df['confidence'], plot_df['inc_time_s'], label='IncMiner', linewidth=2, **get_style('IncMiner'), **inc_style)
            if 'batch_time_s' in plot_df.columns:
                ax.plot(plot_df['confidence'], plot_df['batch_time_s'], label='BatchMiner', linewidth=2, **get_style('BatchMiner'), **batch_style)
            if not args.paper:
                ax.set_xlabel('Confidence')
            ax.set_ylabel('Time (s)')
            if args.paper:
                pass  # No title in paper mode
            else:
                ax.set_title(f'{name} - Time vs Confidence{speedup_str}')
            if not args.paper:
                ax.legend(frameon=True)
            if not args.paper:
                ax.grid(True, alpha=0.3)
            output_path = output_dir / f"{name}_confidence_time.pdf"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            plots_generated.append(output_path)

        # Plot: Recall vs confidence
        if 'confidence' in conf_df.columns and 'inc_recall' in conf_df.columns:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_df = conf_df.sort_values('confidence')
            recall_style = {}
            if not args.paper:
                recall_style = {'marker': 'o', 'color': 'cyan'}
            ax.plot(plot_df['confidence'], plot_df['inc_recall'], linewidth=2, **recall_style)
            if not args.paper:
                ax.set_xlabel('Confidence')
            ax.set_ylabel('Recall')
            if not args.paper:
                ax.set_title(f'{name} - Recall vs Confidence')
            ax.set_ylim([0, 1.05])
            if not args.paper:
                ax.grid(True, alpha=0.3)
            output_path = output_dir / f"{name}_confidence_recall.pdf"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            plots_generated.append(output_path)

    # COUNT-SKETCH-W experiments
    cs_w_df = df[df['experiment_type'] == 'COUNT-SKETCH-W']
    if len(cs_w_df) > 0:
        speedup_str = get_speedup_stats(cs_w_df) if not args.paper else ""
        if 'counter_sketch_w' in cs_w_df.columns and 'inc_time_s' in cs_w_df.columns:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_df = cs_w_df.sort_values('counter_sketch_w')
            cs_style = {}
            if not args.paper:
                cs_style = {'marker': 'o'}
            ax.plot(plot_df['counter_sketch_w'], plot_df['inc_time_s'], linewidth=2, **cs_style)
            if not args.paper:
                ax.set_xlabel('Counter Sketch W')
            ax.set_xscale('log')
            ax.set_ylabel('Time (s)')
            if args.paper:
                pass  # No title in paper mode
            else:
                ax.set_title(f'{name} - Time vs Counter Sketch W{speedup_str}')
            if not args.paper:
                ax.grid(True, alpha=0.3)
            output_path = output_dir / f"{name}_cs_w.pdf"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            plots_generated.append(output_path)

    # COUNT-SKETCH-H experiments
    cs_h_df = df[df['experiment_type'] == 'COUNT-SKETCH-H']
    if len(cs_h_df) > 0:
        speedup_str = get_speedup_stats(cs_h_df) if not args.paper else ""
        if 'counter_sketch_h' in cs_h_df.columns and 'inc_time_s' in cs_h_df.columns:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_df = cs_h_df.sort_values('counter_sketch_h')
            cs_style = {}
            if not args.paper:
                cs_style = {'marker': 'o'}
            ax.plot(plot_df['counter_sketch_h'], plot_df['inc_time_s'], linewidth=2, **cs_style)
            if not args.paper:
                ax.set_xlabel('Counter Sketch H')
            ax.set_ylabel('Time (s)')
            if args.paper:
                pass  # No title in paper mode
            else:
                ax.set_title(f'{name} - Time vs Counter Sketch H{speedup_str}')
            if not args.paper:
                ax.grid(True, alpha=0.3)
            output_path = output_dir / f"{name}_cs_h.pdf"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            plots_generated.append(output_path)

    return plots_generated


def main():
    print("="*80)
    print("Experiment Review")
    print("="*80)
    print(f"Data directory: {DATA_DIR.absolute()}")
    if args.paper:
        print(f"Mode: Paper-style plots in {OUTPUT_DIR.absolute()}")
    else:
        print(f"Mode: Review plots in {OUTPUT_DIR.absolute()}")

    # Load all datasets
    datasets = load_all_csvs(DATA_DIR)

    if not datasets:
        print("No CSV files found!")
        return

    # Summarize each dataset
    for name, df in datasets.items():
        summarize_dataset(name, df)

    # Generate plots
    print(f"\n{'='*80}")
    print("Generating plots...")
    print(f"{'='*80}")

    all_plots = []
    for name, df in datasets.items():
        plots = plot_experiment(name, df, OUTPUT_DIR)
        all_plots.extend(plots)
        if plots:
            print(f"✓ {name}: {len(plots)} plots generated")

    print(f"\n{'='*80}")
    print(f"Summary: {len(datasets)} datasets reviewed, {len(all_plots)} plots generated")
    print(f"Plots saved to: {OUTPUT_DIR.absolute()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

