# Cluster Headache Burden — Monte Carlo Simulation

A quantitative model of the **global burden of cluster headache (CH)**, one of the most
intensely painful conditions known to medicine. The model generates a synthetic worldwide
CH population, simulates every attack each patient has in a year, and aggregates the result
into a global picture of how much time is spent in pain — and at what intensity.

The headline outputs are deliberately concrete:

- a **global distribution of attack peak intensities** (NRS 1–10),
- **raw per-patient `(duration, peak_intensity)` data** for every simulated attack, and
- global **person-years spent in attack**, broken down by the true time spent *at* each
  intensity level (not just by an attack's peak).

Every modelling assumption is an explicit, documented **lever** — there are no hidden magic
numbers. An interactive web UI lets you move each lever and watch the global numbers respond,
including a sensitivity ("tornado") analysis over each parameter's literature-plausible range.

> **Scope note.** This is a generative model of *attacks* (how `(duration, intensity)` tuples
> are produced). It deliberately does **not** include the downstream question of how to *value*
> that pain (e.g. mapping a linear 0–10 scale onto a logarithmic suffering scale). That valuation
> layer is intentionally kept separate.

---

## Quick start

Requires Python 3.10+ and NumPy (the only dependency).

```bash
pip install numpy

# Run the model once and print a summary to the terminal:
python3 ch_simulation.py

# Or launch the interactive web UI (sliders + charts, no extra deps):
python3 server.py            # then open http://localhost:8000
python3 server.py 8080       # custom port
```

To publish the dashboard **publicly with no backend** (it runs the model
in-browser via Pyodide), see [`DEPLOY.md`](DEPLOY.md).

---

## What the model does

Each patient is drawn from distributions and then has a full year of attacks simulated. Each
attack is a tuple `(duration_minutes, peak_intensity)` where `peak_intensity` is an integer 1–10.
Results are scaled to the worldwide CH population via annual prevalence.

**Key modelling decisions** (full reasoning and citations in
[`CH_simulation_sources.md`](CH_simulation_sources.md)):

- **Patients are stratified** by subtype (episodic ≈ 85% / chronic ≈ 15%) and by a *continuous*
  treatment-efficacy parameter — not a hard treated/untreated binary.
- **Intensity is a two-level model.** Each patient has a latent severity (between-patient
  variation); their individual attacks vary around it (within-patient variation). Aggregated
  studies can't separate these, so both are explicit levers.
- **Per-attack peak comes from prospective diaries** (mean ≈ 7/10), *not* the ceiling-loaded
  retrospective "9.7/10, 72% rate it 10/10" figure — that is a recalled *worst-ever* rating,
  not a per-attack measurement.
- **An effective abortive truncates *duration*** (it aborts the attack at the time-to-relief),
  it does **not** lower the peak — the peak (~9 min in) is typically reached before the abortive
  acts. (`treated_peak_intensity_reduction` defaults to 0.)
- **A preventive cuts *frequency*** on a separate channel — fewer attacks per year (shorter
  bouts / lower daily frequency), not shorter or milder individual attacks. Default responder
  rate (0.42) is participant-weighted from the Rusanen 2022 survey (verapamil, corticosteroids,
  lithium, topiramate, melatonin).
- **Global scaling** uses a 1-year prevalence of 53/100,000 adults (Fischera 2007), applied to
  the adult population (~5.8B) → ≈ 3.07M sufferers in a given year.

---

## Repository contents

| File | What it is |
|---|---|
| [`ch_simulation.py`](ch_simulation.py) | The model. `Config` (all levers) → `simulate()` → `SimulationResult` with raw per-attack arrays, the global intensity distribution, true time-at-each-level, CSV export, and a `sweep()` helper for sensitivity. |
| [`webapi.py`](webapi.py) | Shared compute layer (slider metadata, plausible ranges, and `dispatch(endpoint, params)`). No transport code, so the local server and the in-browser build run identical logic. |
| [`server.py`](server.py) | Zero-dependency stdlib HTTP server for local dev; a thin wrapper over `webapi.dispatch`. |
| [`index.html`](index.html) | Browser UI: a slider per lever, live Plotly charts, and a sensitivity tornado. Dual-mode: uses the local server if present, else runs the model in-browser via Pyodide. |
| [`DEPLOY.md`](DEPLOY.md) | How to publish the dashboard as a static page (e.g. `clusterfree.org/burden`) with no backend. |
| [`CH_simulation_sources.md`](CH_simulation_sources.md) | Provenance for **every** default value, tagged `[measured]` / `[pooled]` / `[definitional]` / `[assumption]`, with citations. |
| [`CH simulation parameters.md`](CH%20simulation%20parameters.md) | Distilled research parameters behind the model (frequency, duration, intensity, treatment). |
| [`validate.py`](validate.py) | Sanity-check suite: hard invariants, per-patient physical plausibility, and population realism vs the literature, plus a random-patient eyeball sample. |

---

## Outputs at a glance

`SimulationResult` exposes, among others:

- `patient_attacks(i)` — the raw `(duration, peak_intensity)` list for patient *i*.
- `intensity_distribution(weight=...)` — global attacks/yr by peak intensity (or attack-minutes).
- `time_at_levels()` — global minutes actually spent *at* each intensity level, using a
  within-attack profile (rise ≈ 9 min, decline ≈ 20 min, plateau = remainder).
- `summary()` — global sufferers, attacks/yr, person-years in attack, and person-years/days
  spent at ≥7, ≥9, and 10/10.
- `export_csv()` / `grouped_csv()` — raw per-attack or per-patient data for downstream analysis.

---

## Using it as a library

```python
from ch_simulation import simulate

# Run with defaults, or override any lever:
res = simulate(treatment_access_fraction=0.30, intensity_mean=7.2)

print(res.summary()["global_person_years_at_10"])
print(res.patient_attacks(0)[:5])        # first patient's first 5 attacks

# One-parameter sensitivity sweep:
from ch_simulation import sweep
sweep("abort_prob_mean", [0.5, 0.6, 0.7], metric="global_person_years_in_attack")
```

---

## Cost-effectiveness (in the dashboard)

The dashboard has a **cost-effectiveness panel** that estimates what a nonprofit averts by
helping patients reach treatment — no spreadsheet required. Inputs: annual budget, patients
reached/yr, and an **average effect size**. Outputs: attacks / severe attacks / attack-hours /
**DLES** averted per year, and the corresponding cost-effectiveness ratios ($ per DLES averted,
etc.). It assumes beneficiaries gain *both* acute and preventive treatment and follow the
simulated episodic/chronic mix, and it reflects every simulation lever on the left, via
`cost_effectiveness()` / the `/api/cost_effectiveness` endpoint. (The underlying function still
accepts `channel` and `beneficiary_episodic_share` arguments if you want to model a single
channel or a targeted cohort from Python.)

**Effect size** = the average fraction of the full *no-treatment → full-access* benefit your
beneficiaries actually capture. It absorbs both the new-access/upgrade blend (some patients gain
treatment from scratch, others just upgrade a mediocre one) and additionality. If the effect
size is independent of the patient, only its **mean** affects the totals:

    total_averted = patients_reached × effect_size_mean × mean(full_benefit)
    cost-effectiveness = annual_budget / total_averted

**DLES** (Days Lived in Extreme Suffering) = minutes in ≥9/10 pain ÷ 60 ÷ 24.

### Counterfactual export

For raw per-patient rows, the **"Export CSV"** button produces a spreadsheet-ready file (via
`/api/export_counterfactual` / `counterfactual_csv()`).

Each row is one currently-**untreated** patient, with their baseline attacks plus the model's
counterfactual "with access" attacks under **three interventions** — abortive-only,
preventive-only, and both. The `attacks_with_access_*` columns use the same `(dur,intensity)`
tuple format as the raw export, so a spreadsheet's own DLES / severity formulas can be applied
to each scenario; convenience averted columns (attacks, minutes, severe ≥9, and
`dles_averted_*`) are included too.

Why it beats a flat "X% fewer attacks": access changes the *shape* of the burden, not just the
count. **Abortives truncate duration** (same attack count, shorter attacks), **preventives
remove whole attacks**, and **neither changes the peak**. A single scalar can't represent that —
it typically undercounts time-in-pain and DLES averted by ~2× when abortive access is involved,
because the suffering is dominated by time at ≥9/10, which abortives cut without removing attacks.

Lever changes on the dashboard (efficacy, access, etc.) flow through to the export.

## Validation

[`validate.py`](validate.py) runs a battery of sanity checks and prints a PASS/WARN/FAIL
report (exit code is nonzero if any hard check fails, so it doubles as a CI test):

```bash
python3 validate.py            # default cohort
python3 validate.py 50000      # bigger cohort -> better outlier coverage
python3 validate.py 1000 7     # cohort size, seed
```

- **Invariants** — must hold for any parameters: determinism under a fixed seed, array-length
  consistency, intensity ∈ [1,10], durations within bounds (intrinsic vs aborted floors),
  preventives only *reduce* attack counts, correct global scaling.
- **Physical plausibility** — per-patient outlier scan: nobody in attack more than the whole
  year (a denominator-free impossibility check), nobody >24 h/active-day, attacks/active-day
  respects the ICHD ≤8 ceiling (with a rounding tolerance), nobody pinned to the safety cap.
- **Realism** — population aggregates vs the literature (% episodic ≈85%, mean peak ≈7, median
  duration ≈30–45 min, episodic < chronic frequency, right-skewed distributions, ~3M sufferers).
  These are WARNs, not failures, since they legitimately shift as you move the levers.

**Known modelling simplifications** the checks make explicit: (1) attack frequency and duration
are sampled independently, so the extreme joint tail (very frequent *and* very long) can imply an
implausibly dense day for a handful of patients — negligible for aggregate burden, but reported.
(2) ~1% of episodic patients sample a sub-1-day "bout" yet are floored to ≥1 attack/year (the
annual-prevalence convention that every simulated sufferer is active that year); these are excluded
from per-day density checks.

## Caveats

- This is a **model**, not a measurement. No cohort has recorded a true intra-attack intensity
  time-series, and several treatment-access inputs (especially outside high-income countries)
  are reasoned estimates, not direct data. The provenance file is explicit about which numbers
  are `[measured]` versus `[assumption]`.
- Defaults aim to be defensible central estimates; the point of the lever UI and the sensitivity
  tornado is to make the consequences of each assumption inspectable rather than buried.

## License

No license is currently specified. Contact the repository owner before reuse.
