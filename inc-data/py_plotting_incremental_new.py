#!/usr/bin/env python3
"""
Incremental plotting script converted from py_plotting_incremental_new.ipynb
"""

import matplotlib.pyplot as pyplot
import numpy as np
import os

# pyplot.rc('text', usetex=True)
# pyplot.rc('font', family='serif'
# pyplot.rc('font', family='sans-serif')

# pyplot.rcParams['text.usetex'] = True #Let TeX do the typsetting
# pyplot.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
# pyplot.rcParams['font.family'] = 'sans-serif' # ... for regular text
# pyplot.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here

pyplot.rcParams['text.usetex'] = True
pyplot.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}\sansmath'
pyplot.rcParams['font.family'] = 'sans-serif'
pyplot.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif'

# =============================================================================
# Method names and styling
# =============================================================================

pincminer = "PIncMiner"
batchminer = "BatchMiner"
incdc = "IncDC"
dc3 = "3DC"
incremental = "PIncMiner$_{noCS}$"
pincmineraux = "PIncMiner$_{Aux}$"

# Global style map: label -> {color, marker, markersize} for consistent naming and styling across plots

MARKER_SIZE = 8
FONT_SIZE = 14
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE, titlesize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)

STYLE_MAP = {
    'IncMiner':   {'color': 'C0', 'marker': 'o', 'markersize': MARKER_SIZE},
    'BatchMiner': {'color': 'C1', 'marker': 's', 'markersize': MARKER_SIZE},
    'IncDC':      {'color': 'C2', 'marker': '^', 'markersize': MARKER_SIZE},
    '3DC':        {'color': 'C3', 'marker': 'd', 'markersize': MARKER_SIZE},
    r'PIncMiner$_{noCS}$':  {'color': 'C4', 'marker': 'v', 'markersize': MARKER_SIZE},
    # r'Incremental':  {'color': 'C4', 'marker': 'v', 'markersize': MARKER_SIZE},
    r'PIncMiner$_{noAux}$': {'color': 'C5', 'marker': 'p', 'markersize': MARKER_SIZE},
}

# Wide figure with legend only (shared across all figures)
from matplotlib.lines import Line2D
fig_legend = pyplot.figure(figsize=(6, 0.6))
handles = [Line2D([], [], **{**STYLE_MAP[k], 'linestyle': '-'}, label=k) for k in STYLE_MAP]
fig_legend.legend(handles=handles, ncol=6, loc='center', frameon=True)
pyplot.axis('off')
pyplot.tight_layout()
os.makedirs('plots', exist_ok=True)
os.makedirs('figures', exist_ok=True)
fig_legend.savefig('figures/legend.pdf', bbox_inches='tight')
pyplot.close()

methods = [pincminer, batchminer, incdc, dc3, incremental, pincmineraux]
colors = ['C0', 'C1', 'C2', "C3", "C4", "C5"]
markers = ["o", "s", "^", "d", "v", "p"]

methods_ = [pincminer, batchminer, incdc, dc3]
colors_ = ['green', 'olive', 'brown', 'blue']
markers_ = ["D", "s", "o", ">"]
hatcher_ = ["//\\", "--++", "//", "|--"]

# =============================================================================
# Statistical calculations
# =============================================================================

aaa = np.array([[14.83, 16.54, 26.65, 30.66, 40.91, 9.10, 17.19, 33.84, 41.36, 53.87]])
bbb = np.array([1.42, 2.36, 4.55, 6.63, 8.96, 3.25, 7.06, 26.31, 32.32, 48.69])
print("Average aaa/bbb:", np.average(aaa / bbb), "Max aaa/bbb:", np.max(aaa / bbb))

ccc = np.array([12.98, 13.53, 23.13, 28.60, 32.50, 8.92, 15.19, 31.72, 39.60, 56.68])
print("Average ccc/bbb:", np.average(ccc / bbb), "Max ccc/bbb:", np.max(ccc / bbb))

# print(np.average([98.23, 100.59, 115.76, 119.82, 133.76, 122.61, 115.91, 102.52, 94.15, 81.33]) / np.average([1.42, 2.36, 4.55, 6.63, 8.96, 3.25, 7.06, 26.31, 32.32, 48.69]))

print("Max:", np.max(np.array([98.23, 100.59, 115.76, 119.82, 133.76, 122.61, 115.91, 102.52, 94.15, 81.33]) / np.array([1.42, 2.36, 4.55, 6.63, 8.96, 3.25, 7.06, 26.31, 32.32, 48.69])))

# print(np.average([98.23, 100.59, 115.76, 119.82, 133.76, 122.61, 115.91, 102.52, 94.15, 81.33]) / np.average([1.42, 2.36, 4.55, 6.63, 8.96, 3.25, 7.06, 26.31, 32.32, 48.69]))
print("Average:", np.average(np.array([98.23, 100.59, 115.76, 119.82, 133.76, 122.61, 115.91, 102.52, 94.15, 81.33]) / np.array([1.42, 2.36, 4.55, 6.63, 8.96, 3.25, 7.06, 26.31, 32.32, 48.69])))

print("115.76 / 4.55 =", 115.76 / 4.55)

print("122.61 / 3.25 =", 122.61 / 3.25)

# =============================================================================
# fig-a: NCVoter - Varying Delta D (insertion: 0.01 -> 0.3, deletion: 0.01 -> 0.05)
# =============================================================================

pyplot.figure()
pincminer_ = [1.42, 2.36, 4.55, 6.63, 8.96]
batchminer_ = [98.23, 100.59, 115.76, 119.82, 133.76]

incremental_ = [12.98, 13.53, 23.13, 28.60, 32.50]
pincmineraux_ = [14.83, 16.54, 26.65, 30.66, 40.91]

x_labels = ["(1\\%,1\\%)", "(5\\%,2\\%)", "(15\\%,3\\%)", "(20\\%,4\\%)", "(30\\%,5\\%)"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/ncvoter_varying_deltaD.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-b: DBLP - Varying Delta D (insertion: 0.01 -> 0.05, deletion: 0.01 -> 0.3)
# =============================================================================

print("Max 1:", np.max(np.array([117.77, 128.81, 143.46, 153.17, 167.57]) / np.array([2.74, 7.22, 24.96, 32.59, 51.51])))
print("Max 2:", np.max(np.array([139.40, 136.60, 117.46, 113.00, 101.37]) / np.array([1.35, 15.02, 19.61, 28.87, 38.92])))

pincminer_ = [1.35, 15.02, 19.61, 28.87, 38.92]
batchminer_ = [139.40, 136.60, 117.46, 113.00, 101.37]

incremental_ = [17.43, 44.94, 41.91, 47.42, 54.52]
pincmineraux_ = [21.94, 42.25, 46.81, 47.88, 58.49]

print("Average 1:", np.average(np.array([10.17, 17.32, 37.09, 48.32, 69.03]) / np.array([2.74, 7.22, 24.96, 32.59, 51.51])))
print("Average 2:", np.average(np.array([21.94, 42.25, 46.81, 47.88, 58.49]) / np.array([1.35, 15.02, 19.61, 28.87, 38.92])))
print("1.7 + (2.1 + 4.9) / 2 =", 1.7 + (2.1 + 4.9) / 2)

print("128.81 / 7.22 =", 128.81 / 7.22)
print("136.6 / 15.02 =", 136.6 / 15.02)

print("Average all:", np.average(np.array([117.77, 128.81, 143.46, 153.17, 167.57, 139.40, 136.60, 117.46, 113.00, 101.37]) / np.array([2.74, 7.22, 24.96, 32.59, 51.51, 1.35, 15.02, 19.61, 28.87, 38.92])))
print("Max all:", np.max(np.array([117.77, 128.81, 143.46, 153.17, 167.57, 139.40, 136.60, 117.46, 113.00, 101.37]) / np.array([2.74, 7.22, 24.96, 32.59, 51.51, 1.35, 15.02, 19.61, 28.87, 38.92])))
print("All ratios:", np.array([117.77, 128.81, 143.46, 153.17, 167.57, 139.40, 136.60, 117.46, 113.00, 101.37]) / np.array([2.74, 7.22, 24.96, 32.59, 51.51, 1.35, 15.02, 19.61, 28.87, 38.92]))

pyplot.figure()
pincminer_ = [3.25, 7.06, 26.31, 32.32, 48.69]
batchminer_ = [122.61, 115.91, 102.52, 94.15, 81.33]

incremental_ = [8.92, 15.19, 31.72, 39.60, 56.68]
pincmineraux_ = [9.10, 17.19, 33.84, 41.36, 53.87]

x_labels = ["(1\\%,1\\%)", "(2\\%,5\\%)", "(3\\%,15\\%)", "(4\\%,20\\%)", "(5\\%,30\\%)"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/dblp_varying_deltaD.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-c: DBLP - Varying Delta D^+ (insertion: 0.01 -> 0.3)
# =============================================================================

print("136.60 / 15.02 =", 136.60 / 15.02)

pyplot.figure()
pincminer_ = [2.74, 7.22, 24.96, 32.59, 51.51]
batchminer_ = [117.77, 128.81, 143.46, 153.17, 167.57]

incremental_ = [7.87, 14.81, 35.24, 43.86, 66.42]
pincmineraux_ = [10.17, 17.32, 37.09, 48.32, 69.03]

x_labels = ["1\\%", "5\\%", "15\\%", "20\\%", "30\\%"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/dblp_varying_deltaD+.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-d: NCVoter - Varying Delta D^- (deletion: 0.01 -> 0.3)
# =============================================================================

pyplot.figure()
pincminer_ = [1.35, 15.02, 19.61, 28.87, 38.92]
batchminer_ = [139.40, 136.60, 117.46, 113.00, 101.37]

incremental_ = [17.43, 44.94, 41.91, 47.42, 54.52]
pincmineraux_ = [21.94, 42.25, 46.81, 47.88, 58.49]

x_labels = ["1\\%", "5\\%", "15\\%", "20\\%", "30\\%"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/ncvoter_varying_deltaD-.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# Additional statistics
# =============================================================================

pincminer_ = np.array([4.49, 3.80, 3.14, 2.76, 2.71])
batchminer_ = np.array([113.94, 111.57, 119.35, 86.56, 84.34])
print("Average batch/pinc:", np.average(batchminer_ / pincminer_), "Max batch/pinc:", np.max(batchminer_ / pincminer_))

print("Average 1:", np.average(np.array([832, 1749, 5000]) / np.array([2.78, 21, 70])))
print("Average 2:", np.average(np.array([11, 21, 30, 16, 157, 442]) / np.array([2.78, 21, 70, 2.97, 24.80, 48.65])))
print("Max 1:", np.max(np.array([832, 1749, 5000]) / np.array([2.78, 21, 70])))
print("Max 2:", np.max(np.array([11, 21, 30, 16, 157, 442]) / np.array([2.78, 21, 70, 2.97, 24.80, 48.65])))

# =============================================================================
# fig-e: NCVoter - Varying Delta D+
# =============================================================================

pyplot.figure()
pincminer_ = [0.12, 0.25, 0.42]

incdc_ = [832, 1749, 5000]
dc_ = [11, 21, 30]

x_labels = ["1\\%", "3\\%", "5\\%"]
x = np.arange(3)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, incdc_, label=incdc, marker=markers[2], ms=12, color=colors[2], ls='-', markeredgewidth=4)
pyplot.plot(x, dc_, label=dc3, marker=markers[3], ms=12, color=colors[3], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(3), x_labels)
pyplot.ylim(0.1, 3600)
pyplot.yscale('log')
pyplot.savefig("./figures/ncvoter_dc_varying_deltaD_plus.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-f: NCVoter - Varying Delta D-
# =============================================================================

pyplot.figure()
pincminer_ = [0.13, 0.28, 0.46]
dc_ = [16, 157, 442]

x_labels = ["1\\%", "3\\%", "5\\%"]
x = np.arange(3)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, dc_, label=dc3, marker=markers[3], ms=12, color=colors[3], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(3), x_labels)
pyplot.ylim(0.1, 3600)
pyplot.yscale('log')
pyplot.savefig("./figures/ncvoter_dc_varying_deltaD_minus.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-g: Inspection - Varying support threshold
# =============================================================================

pyplot.figure()
pincminer_ = [4.49, 3.80, 3.14, 2.76, 2.71]
batchminer_ = [113.94, 111.57, 119.35, 86.56, 84.34]

incremental_ = [12.63, 5.27, 3.19, 2.69, 2.67]
pincmineraux_ = [15.56, 7.09, 4.94, 5.06, 5.06]

x_labels = ["10$^{-6}$", "10$^{-5}$", "10$^{-4}$", "10$^{-3}$", "10$^{-2}$"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/inspection_varying_support.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-h: Inspection - Varying confidence threshold
# =============================================================================

pincminer_ = np.array([3.93, 4.04, 3.88, 3.99])
batchminer_ = np.array([109.53, 118.29, 121.83, 116.69])
print("Average batch/pinc:", np.average(batchminer_ / pincminer_), "Max batch/pinc:", np.max(batchminer_ / pincminer_))

pyplot.figure()
pincminer_ = [3.93, 4.04, 3.88, 3.99]
batchminer_ = [109.53, 118.29, 121.83, 116.69]

incremental_ = [11.77, 10.15, 12.90, 12.24]
pincmineraux_ = [15.74, 15.44, 15.95, 14.85]

x_labels = [0.7, 0.8, 0.9, 0.95]
x = np.arange(4)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(4), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/inspection_varying_confidence.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-i: Adult - Varying |D|
# =============================================================================

print("Average ratio:", np.average(np.array([2.11, 2.60, 3.19, 3.73, 4.82, 26.76, 52.85, 76.50, 97.26, 124.30]) / np.array([0.51, 0.68, 0.75, 0.90, 1.05, 2.06, 2.27, 2.53, 2.89, 3.07])))

pyplot.figure()
pincminer_ = [0.72, 0.70, 0.73, 0.74, 0.77]
batchminer_ = [2.11, 2.60, 3.19, 3.73, 4.82]

incremental_ = [1.48, 1.61, 1.95, 2.65, 2.14]
pincmineraux_ = [1.35, 1.49, 1.62, 1.83, 1.74]

x_labels = ["20\\%", "40\\%", "60\\%", "80\\%", "100\\%"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/adult_varying_D.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-j: DBLP - Varying |D|
# =============================================================================

pyplot.figure()
pincminer_ = [2.11, 2.09, 2.14, 2.03, 2.06]
batchminer_ = [26.76, 52.85, 76.50, 97.26, 124.30]

incremental_ = [4.16, 4.50, 7.03, 8.04, 8.76]
pincmineraux_ = [3.11, 4.62, 6.99, 8.71, 9.87]

x_labels = ["20\\%", "40\\%", "60\\%", "80\\%", "100\\%"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/dblp_varying_D.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-k: NCVoter - Varying n
# =============================================================================

print("69.32 / 20.93 =", 69.32 / 20.93)

pyplot.figure()
pincminer_ = [1.27, 0.71, 0.66, 0.65, 0.49]
batchminer_ = [69.32, 40.17, 28.49, 27.58, 20.93]

incremental_ = [9.05, 5.68, 3.78, 3.10, 2.97]
pincmineraux_ = [11.53, 8.51, 7.49, 6.48, 6.16]

x_labels = ["4", "8", "12", "16", "20"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/ncvoter_varying_n.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-l: DBLP - Varying n
# =============================================================================

pyplot.figure()
pincminer_ = [3.46, 1.94, 1.23, 1.29, 1.09]
batchminer_ = [87.38, 53.11, 35.92, 28.47, 24.12]

incremental_ = [9.08, 4.66, 3.73, 3.10, 2.67]
pincmineraux_ = [8.81, 5.04, 3.57, 3.02, 2.54]

x_labels = ["4", "8", "12", "16", "20"]
x = np.arange(5)

pyplot.plot(x, pincminer_, label=pincminer, marker=markers[0], ms=12, color=colors[0], ls='-', markeredgewidth=4)
pyplot.plot(x, batchminer_, label=batchminer, marker=markers[1], ms=12, color=colors[1], ls='-', markeredgewidth=4)
pyplot.plot(x, incremental_, label=incremental, marker=markers[4], ms=12, color=colors[4], ls='-', markeredgewidth=4)
pyplot.plot(x, pincmineraux_, label=pincmineraux, marker=markers[5], ms=12, color=colors[5], ls='-', markeredgewidth=4)

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.yscale('log')
pyplot.savefig("./figures/dblp_varying_n.pdf", dpi=200, bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-m: performance breakdown
# =============================================================================

pyplot.figure()
x_labels = ["1\\%", "5\\%", "15\\%", "20\\%", "30\\%"]
dd = [
    [0.014, 0.016, 0.024, 0.026, 0.033],      # F - Filtering
    [2.48e-6, 4.57e-6, 4.35e-6, 4.48e-6, 5.94e-6],  # S - Sample
    [0.036, 0.064, 0.088, 0.115, 0.124],      # E - Expansion
    [0.15, 0.16, 0.20, 0.19, 0.21],           # R - Retrain
    [0.76, 1.32, 1.57, 1.95, 2.44]            # C - Constant
]
labels = ['F', 'S', 'E', 'R', 'C']
markers_bd = ['P', 'X', 'h', 'H', 'D']
x = np.arange(5)

lines = []
lines.append(pyplot.plot(x, dd[0], label=labels[0], marker=markers_bd[0], ms=12, ls='-', markeredgewidth=4)[0])
lines.append(pyplot.plot(x, dd[1], label=labels[1], marker=markers_bd[1], ms=12, ls='-', markeredgewidth=4)[0])
lines.append(pyplot.plot(x, dd[2], label=labels[2], marker=markers_bd[2], ms=12, ls='-', markeredgewidth=4)[0])
lines.append(pyplot.plot(x, dd[3], label=labels[3], marker=markers_bd[3], ms=12, ls='-', markeredgewidth=4)[0])
lines.append(pyplot.plot(x, dd[4], label=labels[4], marker=markers_bd[4], ms=12, ls='-', markeredgewidth=4)[0])

pyplot.ylabel('Running Time (s)', fontsize=18)
pyplot.yscale('log')
pyplot.tick_params(labelsize=18)
pyplot.xticks(np.arange(5), x_labels)
pyplot.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=18)

pyplot.savefig('figures/dblp_add_breakdown.pdf', bbox_inches='tight')
pyplot.close()

# =============================================================================
# fig-n: Adult, varying w for runtime and memory
# =============================================================================

print("Average:", np.average([195.1361313, 190.3718719, 243.1569366, 331.6322479, 518.3354721, 757.4650497, 1153.025909, 1788.624886, 3071.475677, 5631.618446, 10750.13847, 20986.99469, 41471.67333, 82433.11931, 164352.9857, 328188.4701, 655871.4554, 1311231.513][:9]))

fig = pyplot.figure()
w = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33][:9]
runtime = [0.907, 0.583, 0.710, 0.645, 0.617, 0.502, 0.454, 0.508, 0.531, 0.429, 0.498, 0.529, 0.484, 0.493, 0.435, 0.559, 0.720, 0.748][:9]
memory = [195.1361313, 190.3718719, 243.1569366, 331.6322479, 518.3354721, 757.4650497, 1153.025909, 1788.624886, 3071.475677, 5631.618446, 10750.13847, 20986.99469, 41471.67333, 82433.11931, 164352.9857, 328188.4701, 655871.4554, 1311231.513][:9]

ax1 = fig.add_subplot(111)

color1 = 'tab:blue'
ax1.set_ylabel('Runtime (s)', color=color1, fontsize=18)
line1, = ax1.plot(w, runtime, color=color1, marker='o', label='Runtime', ms=12, ls='-', markeredgewidth=4)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', labelsize=18)

ax2 = ax1.twinx()

color2 = 'tab:red'
ax2.set_ylabel('Memory (MB)', color=color2, fontsize=18)
line2, = ax2.plot(w, memory, color=color2, marker='s', label='Memory', ms=12, ls='-', markeredgewidth=4)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log')

ax1.legend([line1, line2], ['Runtime', 'Memory'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=18)

pyplot.tight_layout()
pyplot.savefig('figures/adult_varying_w.pdf', bbox_inches='tight')
pyplot.close()

# =============================================================================
# figure o: Adult, varying h
# =============================================================================

fig = pyplot.figure()
h = [1, 3, 5, 7, 9]
runtime = [1.52, 1.36, 1.22, 1.19, 1.09]
memory = [147, 163, 185, 198, 214]

ax1 = fig.add_subplot(111)
ax1.set_xticks(h)

color1 = 'tab:blue'
ax1.set_ylabel('Runtime (s)', color=color1, fontsize=18)
line1, = ax1.plot(h, runtime, color=color1, marker='o', label='Runtime', ms=12, ls='-', markeredgewidth=4)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=18)
ax1.tick_params(axis='x', labelsize=18)

ax2 = ax1.twinx()

color2 = 'tab:red'
ax2.set_ylabel('Memory (MB)', color=color2, fontsize=18)
line2, = ax2.plot(h, memory, color=color2, marker='s', label='Memory', ms=12, ls='-', markeredgewidth=4)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=18)

ax1.legend([line1, line2], ['Runtime', 'Memory'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=18)

pyplot.tight_layout()
pyplot.savefig('figures/adult_varying_h.pdf', bbox_inches='tight')
pyplot.close()

# =============================================================================
# figure p: Adult, varying h - Recall
# =============================================================================

pyplot.figure()
h = [1, 3, 5, 7, 9]
actual = [0.975, 0.975, 0.975, 0.975, 0.975]
expected = [0.972817, 0.986044, 0.987786, 0.988464, 0.988825]

line1, = pyplot.plot(h, actual, marker='o', label='Actual', ms=12, ls='-', markeredgewidth=4)
line2, = pyplot.plot(h, expected, marker='s', label='Expected', ms=12, ls='-', markeredgewidth=4)

pyplot.ylabel('Recall', fontsize=18)
pyplot.tick_params(labelsize=18)
pyplot.xticks(h)

pyplot.legend([line1, line2], ['Actual', 'Expected'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=18)
pyplot.ylim(0.90, 1.0)
pyplot.tight_layout()
pyplot.savefig('figures/adult_recall_varying_h.pdf', bbox_inches='tight')
pyplot.close()

print("\nAll figures generated successfully!")
