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

---

## What the model does

Each patient is drawn from distributions and then has a full year of attacks simulated. Each
attack is a tuple `(duration_minutes, peak_intensity)` where `peak_intensity` is an integer 1–10.
Results are scaled to the worldwide CH population via annual prevalence.

**Key modelling decisions** (full reasoning and citations in
[`CH_simulation_sources.md`](CH_simulation_sources.md)):

- **Patients are stratified** by subtype (episodic ≈ 80% / chronic ≈ 20%) and by a *continuous*
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
- **Global scaling** uses a 1-year prevalence of 53/100,000 adults (Fischera 2007), applied to
  the adult population (~5.8B) → ≈ 3.07M sufferers in a given year.

---

## Repository contents

| File | What it is |
|---|---|
| [`ch_simulation.py`](ch_simulation.py) | The model. `Config` (all levers) → `simulate()` → `SimulationResult` with raw per-attack arrays, the global intensity distribution, true time-at-each-level, CSV export, and a `sweep()` helper for sensitivity. |
| [`server.py`](server.py) | Zero-dependency stdlib HTTP server exposing the model as a small JSON API (`/api/simulate`, `/api/sensitivity`, `/api/export`). |
| [`index.html`](index.html) | Browser UI: a slider per lever, live Plotly charts, and a sensitivity tornado over each parameter's literature-plausible range. |
| [`CH_simulation_sources.md`](CH_simulation_sources.md) | Provenance for **every** default value, tagged `[measured]` / `[pooled]` / `[definitional]` / `[assumption]`, with citations. |
| [`CH simulation parameters.md`](CH%20simulation%20parameters.md) | Distilled research parameters behind the model (frequency, duration, intensity, treatment). |

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

## Caveats

- This is a **model**, not a measurement. No cohort has recorded a true intra-attack intensity
  time-series, and several treatment-access inputs (especially outside high-income countries)
  are reasoned estimates, not direct data. The provenance file is explicit about which numbers
  are `[measured]` versus `[assumption]`.
- Defaults aim to be defensible central estimates; the point of the lever UI and the sensitivity
  tornado is to make the consequences of each assumption inspectable rather than buried.

## License

No license is currently specified. Contact the repository owner before reuse.
