# (f)/(g) per-slot seed remine (v306)

Status: **COMPLETE** (2026-09-15)

## Spec
- Image: `harbor.grandhoo.com/rock/piod/incminer-exp:v306`
- Variants: `pincminer,nocs,noaux,naive,staticcorr`
- Seeds: `_fg_batch_seed_probe/SEED_OVERRIDES/{dblp,ncvoter}_{f,g}.txt`
- Jobs: **50/50** (`incminer-exp-ffgrem-*`), `ok=50 fail=0`
- Scratch: `_fg_seed_remine/`
- Archive of prior plot trees: `_archive_fg_pre_seedremine_20260915_154655/`
- Driver: `run_fg_seed_remine.py`
- Log: `_fg_seed_remine_run.log`

## Installed plot paths
- `f_dblp_add_tables/` + `f_dblp_add/`
- `g_dblp_union_delete/` (+ tables)
- `f_ncvoter_add/` + `g_ncvoter_delete_pct5/`

## Figures refreshed
- `exp2026_v7/figure5_mcorr_v283/Figure5_performance_evaluation.png`
- `exp2026_v7/figure5_mcorr_v283/Figure5_ncvoter_fg.png` (+ pdf)

## Batch soft-monotone (remine `batch_runtime_s`)

| curve | 1% | 5% | 15% | 20% | 30% | endpoint |
|------|---:|---:|----:|----:|----:|----------|
| NCVoter f↑ | 183 | 175 | 144 | 127 | 200 | first≤last |
| NCVoter g↓ | 252 | 221 | 178 | 198 | 157 | first≥last |
| DBLP f↑ | 230 | 237 | 276 | 217 | 271 | first≤last |
| DBLP g↓ | 350 | 267 | 303 | 320 | 313 | first≥last |

Note: absolute Batch seconds differ from pinc-only probe (cluster load / 5-variant job), but `SAMPLE_SEED` is confirmed in job logs (e.g. NCVoter add1 → 731). Soft endpoint trends hold; mid-slot dips remain (soft mono).

## Replot only
```bash
python3 run_fg_seed_remine.py --plot-only
```
