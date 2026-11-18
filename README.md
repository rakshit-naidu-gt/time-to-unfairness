# Time-to-Unfairness

Code for *"Time-to-Unfairness: How Fast Does a Deployed Model Go Stale, and
for Whom?"* .

A model that passes a fairness audit at deployment holds its certified
property only as long as the population it was certified on remains
stationary. Under population drift, this property can decay at a different
rate for each demographic group, producing a group-specific validity window
that fixed-schedule compliance audits are not designed to detect. This repo
contains the degradation-rate estimators, the hazard-ratio gate experiment,
and the empirical validation on three ACS PUMS tasks (income, employment,
poverty) described in the paper.

## Layout

```
core/            Equity-ratio / qualification-function library shared across experiments
  erd.py           Equity Ratio Disparity (ERD) metric (Adams' equity theory)
  metrics.py       Demographic parity, equalized odds, accuracy
  qualification.py Qualification-score functions (linear / learned / ensemble)
  models.py        Model wrappers: logistic regression, gradient boosting, MLP (+ ERD regularizer)

idea_t1_gate/    T1-gate experiment: per-group degradation rates and hazard ratios
  gate_experiment.py          Original gate: ACS Income, 2014-2018, GO/PARTIAL/KILL decision
  extended_gate.py            Multi-task (income/employment/coverage/poverty) and
                               extended-year (2014-2022) degradation analysis
  discontinuity_correction.py Trend-plus-level-shift fit correcting the 2020-ACS-gap
                               discontinuity that inflates the naive income-task hazard ratios
  model_class_robustness.py   Reruns the income-task HR under logistic regression and
                               gradient boosting (reviewer robustness check vs. the paper's MLP)
  llm_gate.py                  Same drift measurement using a frozen Llama-3.2 (via Ollama)
                               zero-shot classifier on text-rendered ACS Income records
  results/                     Cached CSV/JSON outputs from the scripts above
  paper/                       Figure-generation script (make_figures.py) and output PDF/PNG figures
```

## Setup

```bash
pip install -r requirements.txt
```

`llm_gate.py` additionally requires a local [Ollama](https://ollama.com) server
running the `llama3.2:latest` model (`ollama pull llama3.2 && ollama serve`).

ACS PUMS data is downloaded on first run via `folktables` and cached under
`idea_t1_gate/acs_income/_cache/`.

## Running the experiments

```bash
cd idea_t1_gate

python gate_experiment.py                  # original 2014-2018 gate
python extended_gate.py                     # all tasks, all years
python extended_gate.py --tier 1            # alternate tasks only, no new downloads
python extended_gate.py --tier 2            # extended income years only (2014-2022)
python discontinuity_correction.py          # 2020-gap-corrected hazard ratios + bootstrap CIs
python model_class_robustness.py            # logistic regression / gradient boosting robustness check
python llm_gate.py                          # LLM (Llama-3.2) zero-shot variant, ~25-40 min
python paper/make_figures.py                # regenerate the paper's figures from results/
```

Each script writes its outputs (degradation trajectories, hazard ratios,
bootstrap confidence intervals) as CSV/JSON under `idea_t1_gate/results/`.

## Citation

This is an anonymized submission; a full citation will be added after review.
