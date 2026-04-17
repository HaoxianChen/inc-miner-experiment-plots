#!/usr/bin/env python3
"""
Calculate speedup numbers for fig-e and fig-f
"""

import numpy as np

print("=" * 70)
print("Speedup Calculations for Fig-e and Fig-f")
print("=" * 70)
print()

# =============================================================================
# Fig-e: NCVoter - Varying Delta D+ (insertions)
# =============================================================================

print("Fig-e: NCVoter - Varying Delta D+")
print("-" * 70)

# Using new data from notebook
pincminer_plus = np.array([0.12, 0.25, 0.42])
incdc_plus = np.array([832, 1749, 5000])
dc3_plus = np.array([11, 21, 30])

speedup_incdc_plus = incdc_plus / pincminer_plus
speedup_dc3_plus = dc3_plus / pincminer_plus

print(f"{'Delta D+':<10} {'PIncMiner':<12} {'3DC':<12} {'IncDC':<12} {'Speedup vs 3DC':<18} {'Speedup vs IncDC':<18}")
print("-" * 82)
for i, label in enumerate(["1%", "3%", "5%"]):
    print(f"{label:<10} {pincminer_plus[i]:<12.2f} {dc3_plus[i]:<12.2f} {incdc_plus[i]:<12.2f} "
          f"{speedup_dc3_plus[i]:<18.1f}x {speedup_incdc_plus[i]:<18.1f}x")

print()
print(f"Average speedup vs 3DC:    {np.average(speedup_dc3_plus):.1f}x")
print(f"Max speedup vs 3DC:        {np.max(speedup_dc3_plus):.1f}x")
print(f"Average speedup vs IncDC:   {np.average(speedup_incdc_plus):.1f}x")
print(f"Max speedup vs IncDC:       {np.max(speedup_incdc_plus):.1f}x")
print()

# =============================================================================
# Fig-f: NCVoter - Varying Delta D- (deletions)
# =============================================================================

print("Fig-f: NCVoter - Varying Delta D-")
print("-" * 70)

# Using new data
pincminer_minus = np.array([0.13, 0.28, 0.46])
dc3_minus = np.array([16, 157, 442])

speedup_dc3_minus = dc3_minus / pincminer_minus

print(f"{'Delta D-':<10} {'PIncMiner':<12} {'3DC':<12} {'Speedup vs 3DC':<18}")
print("-" * 52)
for i, label in enumerate(["1%", "3%", "5%"]):
    print(f"{label:<10} {pincminer_minus[i]:<12.2f} {dc3_minus[i]:<12.2f} {speedup_dc3_minus[i]:<18.1f}x")

print()
print(f"Average speedup vs 3DC:    {np.average(speedup_dc3_minus):.1f}x")
print(f"Max speedup vs 3DC:        {np.max(speedup_dc3_minus):.1f}x")
print()

# =============================================================================
# Combined (fig-e + fig-f)
# =============================================================================

print("Combined (fig-e + fig-f)")
print("-" * 70)

all_speedup_dc3 = np.concatenate([speedup_dc3_plus, speedup_dc3_minus])
all_speedup_incdc = speedup_incdc_plus  # only from fig-e, since IncDC doesn't support deletions

print(f"Overall average speedup vs 3DC:    {np.average(all_speedup_dc3):.1f}x")
print(f"Overall max speedup vs 3DC:        {np.max(all_speedup_dc3):.1f}x")
print(f"Overall average speedup vs IncDC:   {np.average(all_speedup_incdc):.1f}x")
print(f"Overall max speedup vs IncDC:       {np.max(all_speedup_incdc):.1f}x")
print()
print("=" * 70)
print("Paper claims: 4.4x and 151x average, up to 9x and 299x")
print("=" * 70)
