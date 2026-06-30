"""
Zero-dependency web UI for the cluster-headache simulation.

Runs a stdlib HTTP server (no Flask/pip needed -- only numpy, via ch_simulation).
  - GET /            -> serves index.html (sliders + Plotly charts)
  - GET /api/config  -> default Config values + metadata for building the sliders
  - GET /api/simulate?lever=value&... -> runs the sim, returns JSON results

Usage:
    python3 server.py            # then open http://localhost:8000
    python3 server.py 8080       # custom port
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from ch_simulation import Config, simulate, counterfactual_csv, cost_effectiveness

HERE = Path(__file__).parent
INDEX = HERE / "index.html"

# Levers exposed in the UI, with [min, max, step] slider ranges.
SLIDERS = {
    "episodic_fraction":          [0.0, 1.0, 0.01],
    "treatment_access_fraction":  [0.0, 1.0, 0.01],
    "abort_prob_mean":            [0.0, 0.95, 0.01],
    "treat_fraction":             [0.0, 1.0, 0.01],
    "placebo_abort_prob":         [0.0, 0.5, 0.01],
    "aborted_duration_mean_min":  [3.0, 60.0, 1.0],
    "treated_peak_intensity_reduction": [0.0, 0.8, 0.01],
    "preventive_access_fraction": [0.0, 1.0, 0.01],
    "preventive_responder_mean":  [0.0, 0.95, 0.01],
    "preventive_responder_reduction_mean": [0.5, 0.95, 0.01],
    "preventive_responder_reduction_sd":   [0.0, 0.4, 0.01],
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
    "n_patients":                 [2000, 40000, 1000],
}


# Literature-plausible (low, high) range per parameter, for the sensitivity tornado
# -- the real uncertainty, NOT the slider bounds. Source noted per line.
PLAUSIBLE = {
    "annual_prevalence_per_100k": (26.0, 95.0),   # Fischera 2007 95% CI
    "episodic_fraction":          (0.75, 0.90),   # Vikelis 77.5% .. Wei ~90%
    "treatment_access_fraction":  (0.10, 0.30),   # global-access research range
    "abort_prob_mean":            (0.50, 0.78),   # responder: O2 65 / triptan 64 / SC suma 78
    "treat_fraction":             (0.70, 0.95),   # Snoer ~85%
    "placebo_abort_prob":         (0.10, 0.20),   # acute-RCT placebo (suma 17, O2 20)
    "aborted_duration_mean_min":  (7.0, 20.0),    # time-to-relief: suma ~7 .. O2 ~15-20
    "treated_peak_intensity_reduction": (0.0, 0.20),  # default 0; small upside uncertainty
    "preventive_access_fraction": (0.15, 0.55),   # cheap generics .. bounded by dx delay/LMIC gaps
    "preventive_responder_mean":  (0.35, 0.52),   # Rusanen: all-conventional 0.36 .. verap+steroid 0.52
    "preventive_responder_reduction_mean": (0.50, 0.70),  # responder >=50%; mean somewhat above
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


def run_payload(overrides: dict) -> dict:
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


def _coerce(name, raw):
    """Cast a query-string value to the Config field's type."""
    ftype = {f.name: f.type for f in fields(Config)}.get(name)
    if name == "n_patients":
        return int(float(raw))
    return float(raw)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/config":
            self._json({
                "defaults": asdict(Config()),
                "sliders": SLIDERS,
            })
        elif u.path == "/api/simulate":
            try:
                q = parse_qs(u.query)
                overrides = {k: _coerce(k, v[0]) for k, v in q.items()
                             if k in SLIDERS}
                self._json(run_payload(overrides))
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, code=400)
        elif u.path == "/api/sensitivity":
            try:
                q = parse_qs(u.query)
                state = {k: _coerce(k, v[0]) for k, v in q.items() if k in SLIDERS}
                metric = q.get("metric", ["global_person_years_at_10"])[0]
                n = int(float(q.get("n_sens", ["6000"])[0]))
                base_over = {**state, "n_patients": n}
                base_val = simulate(**base_over).summary()[metric]
                levers = []
                for lever, (mn, mx) in PLAUSIBLE.items():
                    lo = simulate(**{**base_over, lever: mn}).summary()[metric]
                    hi = simulate(**{**base_over, lever: mx}).summary()[metric]
                    levers.append({"lever": lever, "low": lo, "high": hi,
                                   "lo_set": mn, "hi_set": mx})
                levers.sort(key=lambda r: abs(r["high"] - r["low"]), reverse=True)
                self._json({"base": base_val, "metric": metric, "levers": levers})
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, code=400)
        elif u.path == "/api/export":
            try:
                q = parse_qs(u.query)
                overrides = {k: _coerce(k, v[0]) for k, v in q.items()
                             if k in SLIDERS}
                # export uses a dedicated sample size (default 3035), overriding
                # the display n_patients.
                overrides["n_patients"] = int(float(q.get("n_export", ["3035"])[0]))
                body = simulate(**overrides).grouped_csv().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition",
                                 'attachment; filename="ch_simulations.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, code=400)
        elif u.path == "/api/cost_effectiveness":
            # end-to-end cost-effectiveness; combines sim levers with org inputs.
            try:
                q = parse_qs(u.query)
                overrides = {k: _coerce(k, v[0]) for k, v in q.items()
                             if k in SLIDERS}
                overrides["n_patients"] = int(float(q.get("ce_n", ["8000"])[0]))
                kw = dict(
                    annual_budget=float(q.get("annual_budget", ["250000"])[0]),
                    patients_reached=float(q.get("patients_reached", ["500"])[0]),
                    effect_size_mean=float(q.get("effect_size_mean", ["0.6"])[0]),
                )
                # Always model both channels reached; beneficiaries follow the
                # simulated episodic/chronic mix (channel="both", share=None).
                self._json(cost_effectiveness(**kw, **overrides))
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, code=400)
        elif u.path == "/api/export_counterfactual":
            # per-untreated-patient baseline + counterfactual 'with access' attacks
            # under abortive-only / preventive-only / both, for cost-effectiveness.
            try:
                q = parse_qs(u.query)
                overrides = {k: _coerce(k, v[0]) for k, v in q.items()
                             if k in SLIDERS}
                overrides["n_patients"] = int(float(q.get("n_export", ["3035"])[0]))
                body = counterfactual_csv(**overrides).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition",
                                 'attachment; filename="ch_counterfactual.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, code=400)
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *a):  # quieter console
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CH simulation UI -> http://localhost:{port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
