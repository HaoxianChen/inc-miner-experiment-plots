#!/usr/bin/env python3
"""
Analyze correlation between AFF (Additional Feature Workload) and IncMiner runtime
"""

# =============================================================================
# Data
# =============================================================================

# Basic workload (70% data)
basic_workload = 3151814982206

# Incremental workloads
increment_labels = ["1%", "5%", "15%", "20%", "30%"]
increment_workloads = [
    3228027420682,  # 1% (71%)
    3603260717024,  # 5% (75%)
    4627439142368,  # 15% (85%)
    5189031630648,  # 20% (90%)
    6407682385578,  # 30% (100%)
]

# IncMiner runtime from figure c (DBLP varying Delta D+)
incminer_runtime = [2.74, 7.22, 24.96, 32.59, 51.51]  # seconds

# =============================================================================
# Compute AFF
# =============================================================================

aff = [w - basic_workload for w in increment_workloads]
print("=" * 70)
print("AFF (Additional Feature Workload) Calculation")
print("=" * 70)
print(f"Basic workload (70% data):  {basic_workload:,}")
print()
for label, workload, a, rt in zip(increment_labels, increment_workloads, aff, incminer_runtime):
    print(f"Increment {label}:")
    print(f"  Total workload:  {workload:>18,}")
    print(f"  AFF:             {a:>18,}")
    print(f"  IncMiner time:   {rt:>14.2f} s")
    print()

# =============================================================================
# Correlation analysis
# =============================================================================

print("=" * 70)
print("Correlation Analysis: AFF vs IncMiner Runtime")
print("=" * 70)

# Manual Pearson correlation calculation
def pearson_correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = (sum((xi - mean_x)**2 for xi in x))**0.5
    denominator_y = (sum((yi - mean_y)**2 for yi in y))**0.5

    return numerator / (denominator_x * denominator_y)

def linear_regression(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x)**2 for xi in x)

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    # R-squared
    y_pred = [slope * xi + intercept for xi in x]
    ss_total = sum((yi - mean_y)**2 for yi in y)
    ss_residual = sum((yi - yp)**2 for yi, yp in zip(y, y_pred))
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

    return slope, intercept, r_squared

r = pearson_correlation(aff, incminer_runtime)
print(f"Pearson correlation coefficient:  r = {r:.4f}")
if abs(r) > 0.9:
    strength = "very strong"
elif abs(r) > 0.7:
    strength = "strong"
elif abs(r) > 0.5:
    strength = "moderate"
elif abs(r) > 0.3:
    strength = "weak"
else:
    strength = "very weak or no"
direction = "positive" if r > 0 else "negative"
print(f"  -> This indicates a {strength} {direction} correlation")
print()

slope, intercept, r_squared = linear_regression(aff, incminer_runtime)
print(f"Linear regression:")
print(f"  Slope:       {slope:.2e} seconds per unit AFF")
print(f"  Intercept:   {intercept:.2f} seconds")
print(f"  R-squared:   {r_squared:.4f}")
print(f"  Equation:    y = {slope:.2e} * x + {intercept:.2f}")
print()
print(f"  R-squared of {r_squared:.1%} indicates that {r_squared:.1%} of the variance")
print(f"  in IncMiner runtime can be explained by AFF.")
print()

print("=" * 70)
print("Conclusion")
print("=" * 70)
if r_squared > 0.9:
    print("✓ AFF is strongly correlated with IncMiner runtime.")
    print("  The additional feature workload is a very good predictor of")
    print("  the incremental algorithm's performance.")
elif r_squared > 0.7:
    print("✓ AFF is well correlated with IncMiner runtime.")
    print("  The additional feature workload is a good predictor of")
    print("  the incremental algorithm's performance.")
elif r_squared > 0.5:
    print("✓ AFF shows moderate correlation with IncMiner runtime.")
    print("  Other factors also contribute significantly.")
else:
    print("✗ AFF shows limited correlation with IncMiner runtime.")
    print("  Other factors dominate the performance.")
print()
