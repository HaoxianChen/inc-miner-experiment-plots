
# Experiment Review Script

Generates plots from incremental rule discovery experiment results.

## Quick Start

### Generate review plots (with labels/titles/legends):
```bash
python3 review_experiments.py
```
Plots saved to: `review_plots/`

### Generate paper plots (minimal styling):
```bash
python3 review_experiments.py --paper
```
Plots saved to: `plots-paper/`

## What it does

- Loads all CSV files from `inc_vs_batch_v4/`
- Skips AMiner and dblp_ml datasets
- Prints summary statistics and speedup analysis
- Generates plots for:
  - ADD/DELETE experiments (time vs increment ratio, recall)
  - SCALE experiments (time vs scale factor)
  - SUPPORT experiments (time/recall vs support, log scale)
  - CONFIDENCE experiments (time/recall vs confidence)
  - COUNT-SKETCH experiments (time vs W/H parameters)

## Data Sources

- Experimental data: `inc_vs_batch_v4/experiment_results_*.csv`
- Plots output: `review_plots/` or `plots-paper/`

