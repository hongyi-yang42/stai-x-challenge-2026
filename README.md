# STAI-X Challenge 2026 — overdose ED forecasting

My entry for the [STAI-X Challenge 2026](https://www.kaggle.com/competitions/stai-x-challenge-2026), a Kaggle competition forecasting US state-level rates of suspected nonfatal drug overdose ED visits (CDC DOSE-SYS syndromic surveillance).

**Result:** rank 26 / 35, private MAE 0.905, public 0.918. Submitted ~6 hours past the deadline — postmortem in [`CLAUDE.md`](./CLAUDE.md#late-submission-note).

## Quickstart

```bash
pip install -r requirements.txt
# put competition data under ./data/
python solution_portable.py
```

Outputs `submission.csv` (918 rows) at the working directory.

## Approach

One LightGBM per scoring category (`all_drugs`, `all_opioids`, `all_stimulants`), `objective='mae'`. Panel data of 51 jurisdictions × 77 reporting periods.

Main ingredients:

- Fold-isolated jurisdiction × category target encoding (mean / median / log-mean), out-of-fold only
- Google Trends *ratios* (`ratio_fent_od`, `ratio_nalo_od`, …) — raw gtrends are multicollinear; ratios carry the within-state signal
- Within-state z-scores and period-level national-wave anchors
- Recency-weighted drift baseline (EW decay 0.90) blended with the model as a safety net for extrapolation periods

### Things tried but not shipped

- **Image features** (`mat_density` PNGs): pixel-identical per state across all 77 periods, so they're redundant with the `jurisdiction` ID. Dropped.
- **Text features** (`state_doh_release`): keyword counts, release-type classifiers, regex extracts. Engineered but ablation showed no CV lift over gtrends. Left in the code.
- **Ridge / CatBoost / stacking**: on the roadmap, not validated under period-blocked CV in time.

## Repo layout

```
solution_portable.py    # the STAI-X model, single file, ~586 lines
CLAUDE.md               # playbook, reproducibility notes, postmortem
dry_run_setup.py        # fabricates regression + classification datasets
dry_run_validate.py     # runs the domain-agnostic pipeline on them
data/                   # competition data goes here (not committed)
.claude/                # config for the Award-B agent pipeline
requirements.txt
LICENSE
```

The `dry_run_*.py` scripts are a domain-agnostic version of the pipeline built for the competition's Award B track (Claude Code automation). They ingest any tabular CSV + `sample_submission.csv` and produce a valid submission.

## Future work

- Validate Ridge + CatBoost stacked with LightGBM under period-blocked CV
- Re-test text features specifically against the `all_drugs` block
- Factor the gtrends columns (5 → 2 latent factors: period intensity × jurisdiction scale) and model the decomposition explicitly

## License

MIT.
