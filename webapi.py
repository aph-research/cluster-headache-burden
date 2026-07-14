"""
Shared compute layer for the cluster-headache dashboard.

This module contains ALL of the request-handling / payload-building logic, with
no HTTP or transport code. It is imported by:

  * server.py            -> wraps `dispatch()` in a stdlib HTTP server (local dev)
  * index.html (Pyodide) -> calls `dispatch()` directly in the browser (static
                            hosting, e.g. clusterfree.org/burden)

Keeping the logic here means the local server and the in-browser build run the
EXACT same code. `dispatch(endpoint, params)` takes an endpoint name and a flat
dict of string/number params (the query string, essentially) and returns either
a JSON-able dict or, for the CSV endpoints, a CSV string.

Depends only on numpy + ch_simulation (both available under Pyodide).
"""

from __future__ import annotations

from dataclasses import asdict, fields

import numpy as np

from ch_simulation import (
    Config, simulate, counterfactual_csv, cost_effectiveness,
    cost_effectiveness_bands, cost_effectiveness_funnel,
)

# Levers exposed in the UI, with [min, max, step] slider ranges.
SLIDERS = {
    "episodic_fraction":          [0.0, 1.0, 0.01],
    "treatment_access_fraction":  [0.0, 1.0, 0.01],
    "abort_prob_mean":            [0.0, 0.95, 0.01],
    "treat_fraction":             [0.0, 1.0, 0.01],
    "aborted_duration_mean_min":  [3.0, 60.0, 1.0],
    "treated_peak_intensity_reduction": [0.0, 0.8, 0.01],
    "preventive_access_fraction": [0.0, 1.0, 0.01],
    "preventive_responder_mean":  [0.0, 0.95, 0.01],
    "preventive_responder_reduction_mean": [0.5, 0.95, 0.01],
    "preventive_responder_reduction_sd":   [0.0, 0.4, 0.01],
    "preventive_peak_intensity_reduction": [0.0, 0.8, 0.01],
    "intensity_mean":             [3.0, 9.0, 0.1],
    "intensity_between_sd":       [0.0, 3.0, 0.1],
    "intensity_within_sd":        [0.0, 3.0, 0.1],
    "duration_median_min":        [10.0, 120.0, 1.0],
    "duration_sigma":             [0.1, 1.5, 0.05],
    "dur_intensity_slope":        [0.0, 0.15, 0.01],
    "profile_rise_min":           [0.0, 30.0, 1.0],
    "profile_decline_min":        [0.0, 60.0, 1.0],
    "e_attacks_per_day_mean":     [0.5, 8.0, 0.1],
    "c_attacks_per_day_mean":     [0.5, 8.0, 0.1],
    "e_bout_weeks_mean":          [1.0, 26.0, 0.5],
    "e_bouts_per_year_mean":      [0.3, 4.0, 0.1],
    "annual_prevalence_per_100k": [10.0, 150.0, 1.0],
    "n_patients":                 [2000, 16000, 1000],
}


# Literature-plausible (low, high) range per parameter, for the sensitivity tornado
# -- the real uncertainty, NOT the slider bounds. Source noted per line.
PLAUSIBLE = {
    "annual_prevalence_per_100k": (26.0, 95.0),   # Fischera 2007 95% CI
    "episodic_fraction":          (0.75, 0.90),   # Vikelis 77.5% .. Wei ~90%
    "treatment_access_fraction":  (0.10, 0.30),   # global-access research range
    "abort_prob_mean":            (0.50, 0.78),   # responder: O2 65 / triptan 64 / SC suma 78
    "treat_fraction":             (0.70, 0.95),   # Snoer ~85%
    "aborted_duration_mean_min":  (7.0, 20.0),    # time-to-relief: suma ~7 .. O2 ~15-20
    "treated_peak_intensity_reduction": (0.0, 0.20),  # default 0.1; peak mostly precedes abortive
    "preventive_access_fraction": (0.15, 0.55),   # cheap generics .. bounded by dx delay/LMIC gaps
    "preventive_responder_mean":  (0.35, 0.52),   # Rusanen: all-conventional 0.36 .. verap+steroid 0.52
    "preventive_responder_reduction_mean": (0.50, 0.70),  # responder >=50%; mean somewhat above
    "preventive_peak_intensity_reduction": (0.0, 0.25),  # default 0.1; weakly-quantified severity effect
    "intensity_mean":             (6.3, 7.5),     # Russell-rescaled ~6.9 .. Snoer 7.0(+)
    "intensity_between_sd":       (1.0, 2.2),     # assumption; Snoer "low within" => between sizable
    "intensity_within_sd":        (0.5, 1.6),     # assumption; Snoer low within-patient variability
    "duration_median_min":        (35.0, 60.0),   # Hagedorn ~39 / Goebel mode 30 .. survey-skewed
    "duration_sigma":             (0.5, 0.9),     # right-tail shape uncertainty
    "dur_intensity_slope":        (0.0, 0.12),    # 0 = none .. Hagedorn ~0.106
    "e_attacks_per_day_mean":     (1.0, 3.1),     # Vikelis median 1 .. Gaul 3.1
    "c_attacks_per_day_mean":     (1.5, 3.3),     # Vikelis median 2 .. Gaul 3.3
    "e_bout_weeks_mean":          (4.0, 12.0),    # Vikelis 4-6 / review 6-12 / Gaul 8.5
    "e_bouts_per_year_mean":      (1.0, 1.5),     # most ~1 / Gaul 1.2
    "profile_rise_min":           (5.0, 15.0),    # Torelli time-to-peak ~9
    "profile_decline_min":        (15.0, 30.0),   # Snoer resolution ~20
}


def _coerce(name, raw):
    """Cast a query-string value to the Config field's type."""
    if name == "n_patients":
        return int(float(raw))
    return float(raw)


def _overrides(params):
    """Pull just the lever params, coerced to numbers."""
    return {k: _coerce(k, params[k]) for k in params if k in SLIDERS}


def _log_hist(x, nbins=24):
    x = np.asarray(x, float)
    lo, hi = max(1, x.min()), x.max()
    edges = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi + 1), nbins)))
    counts, edges = np.histogram(x, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return {"centers": centers.tolist(), "counts": counts.tolist()}


def _burden_by_group(res):
    """Mean hours/yr spent in attack per patient, by group. Varies across all four
    groups (chronic > episodic via frequency; no-access > access via duration).
    Also returns mean peak intensity per group (which is ~equal by design)."""
    n = len(res.is_episodic)
    pat_min = np.bincount(res.patient_idx, weights=res.duration, minlength=n)
    pat_hours = pat_min / 60.0
    groups = {
        "episodic": res.is_episodic, "chronic": ~res.is_episodic,
        "with access": res.has_access, "no access": ~res.has_access,
        "on preventive": res.on_preventive, "no preventive": ~res.on_preventive,
    }
    a_ep = res.is_episodic[res.patient_idx]
    a_acc = res.has_access[res.patient_idx]
    a_prev = res.on_preventive[res.patient_idx]
    amasks = {"episodic": a_ep, "chronic": ~a_ep,
              "with access": a_acc, "no access": ~a_acc,
              "on preventive": a_prev, "no preventive": ~a_prev}
    return {
        "hours": {g: (float(pat_hours[m].mean()) if m.any() else None)
                  for g, m in groups.items()},
        "intensity": {g: (float(res.intensity[m].mean()) if m.any() else None)
                      for g, m in amasks.items()},
    }


def run_payload(overrides):
    res = simulate(**overrides)
    return {
        "summary": res.summary(),
        "by_peak": res.intensity_distribution("attacks").tolist(),
        "time_at_levels": res.time_at_levels(mode="minutes").tolist(),
        "attacks_pct": res.attacks_per_year_percentiles(),
        # log-binned attacks/yr histogram for the long-tail chart
        "attacks_hist": _log_hist(res.n_attacks),
        # per-group burden: hours/yr in attack per patient (varies by BOTH subtype
        # and access; peak intensity does NOT vary by group, by design).
        "burden_by_group": _burden_by_group(res),
        "example_patient": res.patient_attacks(int(np.argmax(res.n_attacks > 0)))[:30],
    }


def sensitivity_payload(state, metric, n_sens):
    base_over = {**state, "n_patients": n_sens}
    base_val = simulate(**base_over).summary()[metric]
    levers = []
    for lever, (mn, mx) in PLAUSIBLE.items():
        lo = simulate(**{**base_over, lever: mn}).summary()[metric]
        hi = simulate(**{**base_over, lever: mx}).summary()[metric]
        levers.append({"lever": lever, "low": lo, "high": hi,
                       "lo_set": mn, "hi_set": mx})
    levers.sort(key=lambda r: abs(r["high"] - r["low"]), reverse=True)
    return {"base": base_val, "metric": metric, "levers": levers}


def dispatch(endpoint, params=None):
    """Route an endpoint name + params dict to a result.

    Returns a JSON-able dict for most endpoints, or a CSV string for the two
    export endpoints. Raises on bad input (callers translate to an error).
    """
    params = dict(params or {})

    if endpoint == "config":
        return {"defaults": asdict(Config()), "sliders": SLIDERS}

    if endpoint == "simulate":
        return run_payload(_overrides(params))

    if endpoint == "sensitivity":
        metric = params.get("metric", "global_person_years_at_ge9")
        n_sens = int(float(params.get("n_sens", 6000)))
        return sensitivity_payload(_overrides(params), metric, n_sens)

    if endpoint == "cost_effectiveness":
        # ClusterFree: effect size is a truncated-normal (median, sd) -> MC bands.
        overrides = _overrides(params)
        overrides["n_patients"] = int(float(params.get("ce_n", 4000)))
        kw = dict(
            annual_budget=float(params.get("annual_budget", 100000)),
            patients_reached=float(params.get("patients_reached", 500)),
            effect_size_mean=float(params.get("effect_size_mean", 0.6)),
            effect_size_sd=float(params.get("effect_size_sd", 0.15)),
            n_mc=int(float(params.get("n_mc", 5000))),
        )
        # Always model both channels reached; beneficiaries follow the simulated
        # episodic/chronic mix (channel="both", share=None).
        return cost_effectiveness_bands(**kw, **overrides)

    if endpoint == "cost_effectiveness_funnel":
        overrides = _overrides(params)
        overrides["n_patients"] = int(float(params.get("ce_n", 4000)))

        def gauss(name, dmean, dsd):  # (median, sd) truncated-normal factor
            return (float(params.get(f"{name}_mean", dmean)),
                    float(params.get(f"{name}_sd", dsd)))

        kw = dict(
            annual_budget=float(params.get("annual_budget", 100000)),
            annual_unique_visitors=float(params.get("annual_unique_visitors", 10000)),
            patient_fraction=gauss("patient_fraction", 0.75, 0.10),
            engaged_fraction=gauss("engaged_fraction", 0.30, 0.10),
            adoption_fraction=gauss("adoption_fraction", 0.25, 0.10),
            counterfactual_share=gauss("counterfactual_share", 0.50, 0.12),
            clinical_capture=gauss("clinical_capture", 0.50, 0.13),
            n_mc=int(float(params.get("n_mc", 5000))),
        )
        return cost_effectiveness_funnel(**kw, **overrides)

    if endpoint == "export_counterfactual":
        overrides = _overrides(params)
        overrides["n_patients"] = int(float(params.get("n_export", 3035)))
        return counterfactual_csv(**overrides)

    if endpoint == "export":
        overrides = _overrides(params)
        overrides["n_patients"] = int(float(params.get("n_export", 3035)))
        return simulate(**overrides).grouped_csv()

    raise ValueError(f"unknown endpoint: {endpoint}")
