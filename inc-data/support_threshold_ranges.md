# Support threshold ranges by data source

Summary of the `support` column (minimum support threshold) across experiment result CSVs.

---

## Data sources used in the notebook

### vs_incdc (Adult, Airports — vs IncDC/3DC)
| File | Support range | Notes |
|------|---------------|--------|
| `vs_incdc/experiment_results_adult.csv` | **2** (single value) | Same for ADD, DELETE, SCALE |
| `vs_incdc/experiment_results_airports.csv` | **2** (single value) | Same for ADD, DELETE, SCALE |

### inc_vs_batch2 (Hospital, Ncvoter, DBLP_agg — IncMiner vs BatchMiner)
| File | Support range | Notes |
|------|---------------|--------|
| `inc_vs_batch2/experiment_results_hospital_ml.csv` | **1** (TRAIN), **13,206** (ADD, DELETE) | Two levels |
| `inc_vs_batch2/experiment_results_ncvoter_ml.csv` | **282** (TRAIN), **2,827,832** (ADD, DELETE, MACHINES) | Two levels |
| `inc_vs_batch2/experiment_results_dblp_agg.csv` | **11** (TRAIN) → **117,296** (ADD/DELETE/SCALE/etc.), plus **SUPPORT** sweep: 117296, 1172966, 11729666, 117296660, 1172966601 | Main ADD/DELETE use support **117,296**; SUPPORT experiment varies support over 5 orders of magnitude |

---

## Other inc_vs_batch2 files (reference)

| File | Support range |
|------|---------------|
| `experiment_results_adult_ml.csv` | 0, 212, 424, 636, 848, 1060 |
| `experiment_results_inspection_ml.csv` | 4, 48814, 488144, 4881448, 48814483, 488144836 |
| `experiment_results_AMiner_Author_ml.csv` | 11, 23459, 46918, 70377, 93837, 117296, 1172966, 11729666, 117296660, 1172966601 |
| `experiment_results_AMiner_Paper_ml.csv` | 17, 35023, 70047, 105070, 140094, 175117, 1751179, 17511797, 175117977, 1751179778 |
| `experiment_results_AMiner_Author2Paper_ml.csv` | 107, 215737, 431475, 647212, 862950, 1078687, 10786878, 107868788, 1078687882, 10786878828 |

---

## Root (align_dc) — not currently used for plots
| File | Support values |
|------|-----------------|
| `experiment_results_inc_batch_align_dc_adult.csv` | 212, 424, 636, 848, 1060 (scale + ADD) |
| `experiment_results_inc_batch_align_dc_airports.csv` | 661, 1323, 1984, 2646, 3308 |

---

## DBLP_agg detail

- **ADD / DELETE / SCALE / CONFIDENCE / MACHINES**: support = **117,296** (fixed).
- **TRAIN**: support = **11** (tiny sample).
- **SUPPORT** (support-sensitivity experiment): support takes values **117296**, **1172966**, **11729666**, **117296660**, **1172966601** (≈ 10×, 100×, 1000×, 10000× base).

So for the ADD and DELETE plots you use from DBLP_agg, the support threshold is **117,296** for all points.
