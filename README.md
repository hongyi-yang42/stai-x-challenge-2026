# STAI-X Challenge 2026 — State-Level Overdose ED Forecasting

> Personal portfolio writeup. Not a live competition submission (entered as **Late Submission**, Jun 15 2026).

## What this is

**TL;DR.** An entry for [STAI-X Challenge 2026](https://www.kaggle.com/competitions/stai-x-challenge-2026) — a Kaggle competition forecasting state-level suspected nonfatal overdose ED visit rates, mirroring CDC's DOSE-SYS syndromic surveillance system. Final placement: **rank #26 of 35** on the public leaderboard, **private score 0.905** (block-averaged MAE; lower is better) vs **public 0.918**. Submitted 5h47m past the deadline (Late Submission; see [Late submission note](#late-submission-note) in CLAUDE.md).

**What it does.** Per-scoring-category LightGBM (`objective='mae'`) over panel data of 51 jurisdictions × 77 periods. Each category model is trained with fold-isolated jurisdiction target encoding (`jur_cat_mean`, `jur_cat_median`, `jur_cat_log_mean`) — out-of-fold predictions only, no leakage from inference rows. A recency-weighted drift baseline (EW decay 0.90, last-N periods) is computed from train and blended with the model for safety on extrapolation-heavy periods. Features: Google Trends ratios (`ratio_fent_od`, `ratio_nalo_od` etc — period-level ratios carry the real within-state signal, raw gtrends are multicollinear), within-state z-scores, period-level national-wave anchors, covariate imputation defaults derived from train medians.

**What it doesn't do.** No external pre-trained models (no HuggingFace/Kaggle Hub weights, no foundation models). No image features — the `mat_density` PNGs are pixel-identical per jurisdiction across all 77 periods (verified), providing zero information beyond the `jurisdiction` categorical. Text features (`state_doh_release`) are engineered in the code — keyword counts, release-type classification, numeric regex extraction — but extensive ablation showed **zero CV improvement** beyond the LightGBM + gtrends baseline (signal is real but redundant with gtrends). They remain in the codebase for transparency. No stacking across model families.

**Honest result.** We landed at **#26/35** with private MAE 0.905; the top team scored **0.699**. The 0.2-point gap to first place was likely closed by some combination of (a) external signals we couldn't use (competition forbids external data, but pre-trained weights were allowed and we didn't leverage them), (b) stacking across diverse base learners (we used LightGBM only — Ridge/CatBoost variants were on the roadmap but not validated in time), or (c) text features we inspected but didn't have time to validate under the dual-CV gate (period-blocked + forward holdout). The code is honest about what worked and what didn't; the comments document every ablation verdict.

## The Award-B pipeline component

This repo also contains a **domain-agnostic AI pipeline** (the `dry_run_*.py` scripts and `CLAUDE.md` playbook) that was originally developed for [Award B — AI Automation](https://www.kaggle.com/competitions/stai-x-challenge-2026#Awards). Award B required a domain-agnostic agent that organizers would re-run with `claude --dangerously-skip-permissions` on held-out data from a *different* domain. The pipeline:

1. Inventories and joins the data files in `data/`
2. Identifies the target variable and task type (regression or classification) from `sample_submission.csv`
3. Performs quick EDA with leakage checks
4. Builds a validated LightGBM model with 5-fold CV (grouped if a grouping column exists)
5. Produces a correctly formatted `submission.csv`
6. Generates a minimal `report.pdf` with methodology and findings

**The pipeline is explicitly domain-agnostic.** It contains no assumptions about column names, feature types, or modeling choices beyond general tabular ML best practices. Validated on two fabricated datasets (`dry_run_regression/`, `dry_run_classification/`).

To dry-run the agent pipeline on fabricated data:

```bash
python3 dry_run_setup.py        # generates test datasets in dry_run_{reg,classification}/data/
python3 dry_run_validate.py     # runs the full pipeline end-to-end on both
```

## Pipeline at a glance

```
data/ → Inventory → EDA → Baseline Submission → Model (LightGBM + CV) → Final Submission + Report
                          (always produce valid output early)            (improve if budget remains)
```

## Key design decisions

- **Baseline-first**: Always ships a valid submission from simple group means before any complex modeling. Prevents zero-output failures.
- **LightGBM default**: Fast, handles categoricals/missing values natively, strong on tabular data without tuning.
- **Leakage-aware**: All target-derived features (`jur_cat_mean` etc.) computed from training data only, mapped to test via join keys. Fold-isolated.
- **Budget-aware**: Time-boxed phases. If budget runs low, the baseline submission is already written.
- **Schema-portable**: No hardcoded column names, row counts, period IDs, or file paths. Everything derived from the data at runtime. (The Kaggle mount path quirk is handled by `_detect_data_dir()` with a glob fallback — see [Reproducibility notes](CLAUDE.md#reproducibility-notes).)

## Repo structure

```
.
├── CLAUDE.md              # Autonomous playbook + reproducibility notes (the agent reads this)
├── README.md              # This file
├── LICENSE                # MIT License
├── requirements.txt       # Pinned Python dependencies
├── solution_portable.py   # The actual STAI-X model (586 lines, self-contained)
├── data/
│   └── .gitkeep           # Grader drops evaluation data here (currently empty)
├── dry_run_setup.py       # Fabricates test datasets for Award-B pipeline validation
├── dry_run_validate.py    # End-to-end agent pipeline dry-run
└── .claude/
    ├── settings.json      # Permission allowlist for the agent
    └── skills.md          # Baseline-first skill definition
```

## Future work

- Validate Ridge / CatBoost base learners under the dual-CV gate (period-blocked + forward holdout) and stack with LightGBM.
- Re-test drug-specific text features (drug-relevant keywords + release types) under the dual-CV gate. The previous text ablation was stimulant-focused; drug-relevant keywords may carry different signal for the dominant `all_drugs` block.
- Investigate the gtrends decomposition: 5 columns decompose into 2 latent factors (period base intensity × jurisdiction scale factor). A model that explicitly learns this factorization might generalize better under Stage-2 distribution shift.
- The within-state temporal signal (21-33% of target variance) is the bottleneck. Period-level national gtrends wave anchors showed forward-holdout improvement in CV but was not validated before deadline.

## Team

- **Team:** harisheldon42
- **GitHub:** https://github.com/hongyi-yang42/stai-x-challenge-2026

## License

MIT License — see [LICENSE](LICENSE).
