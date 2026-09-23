# WQRegressors — working instructions

## Scope: which model families count

The supported model families, and the only ones any task concerns by default, are:

| Family | Trained by | Input uncertainty |
| --- | --- | --- |
| Gaussian process | `e_Train.train_gp_regressor_model` | analytic, inside the kernel |
| Multiple linear regression | `utils/mlr.py` | via Monte Carlo replicates |
| XGBoost | `e_Train` | via Monte Carlo replicates |
| Transformer | `e_Train` | via Monte Carlo replicates |

**Spikes — excluded unless named explicitly:**

- the **recurrent transformer** and the **LSTM** (`src/e_TrainRecurrent.py`, and the
  `config_recurrent_transformer_*.yml` / `config_lstm_*.yml` that `d_RunResample.py` emits)
- the **particle filter** (`src/n_ParticleFilter.py`, and the `pf_ml`, `pf_mlr`,
  `pf_mlr_single` output directories)

These are alternative approaches that were not successful. They are not described in the
manuscript and are **not kept up to date with other project changes** — do not assume they
run against current code. Their source and outputs are retained for possible future
development, and nothing here asks for them to be deleted.

Do not include a spike in analysis, refactoring, comparisons, verification or option lists,
and do not offer one as a choice, unless the request names it. Note that the code is
misleading on this point: `e_TrainRecurrent.py` describes both recurrent models as "supported
model types" and their configs are generated beside the real ones. **This file is the
authority, not the code.**

## Results series

`data/output/` is gitignored, so nothing in it is version-controlled or recoverable. The
registry of every results series — what produced it, what reads it, what supersedes it — is
`docs/RESULTS_SERIES_REGISTRY.md`. A new series needs a row there in the same commit.

A dataset that is a source for anything currently in the manuscript is retained **in full**
for traceability, and released only once a different, completed dataset supersedes it as that
source.

## Running things

- Python is the local virtual environment: `.venv/Scripts/python`. Never use a system Python,
  and never install packages — if an import fails, stop and report it.
- `docs/RUN_OUTPUT_MANIFEST.md` holds the full run sequence and the gates that must pass
  before results are reportable.
- The LaTeX manuscript is a **sibling repository**, not a submodule, reached through
  `WQ_MANUSCRIPT_ROOT`. Scripts that write into it fail loudly when it is unset.
