"""
Sanity / plausibility checks for the cluster-headache simulation.

Three tiers:
  1. INVARIANTS   -- properties that must hold for ANY parameter set (a FAIL is a
                     real bug). Determinism, array consistency, value bounds,
                     preventives-only-reduce, correct global scaling.
  2. PHYSICAL     -- per-patient plausibility. Nobody can be "in attack" >24h/day
                     during their active period; attacks/active-day must respect the
                     ICHD <=8 ceiling; flag anyone pinned to the safety cap.
  3. REALISM      -- population aggregates vs the literature (drift -> WARN, not a
                     hard failure, since these depend on the chosen levers).

Plus an EYEBALL sample: prints a handful of random simulated patients so you can
gut-check that individual patients look like real cluster-headache patients.

Run:
    python3 validate.py            # default cohort
    python3 validate.py 50000      # bigger cohort -> better outlier coverage
    python3 validate.py 1000 7     # cohort size, seed

Exit code is nonzero if any INVARIANT or PHYSICAL check FAILs (CI-friendly).
"""

from __future__ import annotations

import sys
import numpy as np

from ch_simulation import Config, simulate

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_MARK = {PASS: "\033[32m[PASS]\033[0m", WARN: "\033[33m[WARN]\033[0m",
         FAIL: "\033[31m[FAIL]\033[0m"}


class Report:
    """Collects (tier, status, name, detail) rows and prints a grouped summary."""

    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, tier, status, name, detail=""):
        self.rows.append((tier, status, name, detail))

    def check(self, tier, name, ok, detail="", warn_only=False):
        """ok True -> PASS; ok False -> FAIL (or WARN if warn_only)."""
        status = PASS if ok else (WARN if warn_only else FAIL)
        self.add(tier, status, name, detail)
        return ok

    def range(self, tier, name, value, lo, hi, fmt="{:.3f}", warn_only=True):
        """PASS iff lo <= value <= hi; reports the value and target range."""
        ok = lo <= value <= hi
        detail = f"{fmt.format(value)} (expect {fmt.format(lo)}..{fmt.format(hi)})"
        return self.check(tier, name, ok, detail, warn_only=warn_only)

    def print(self):
        worst = PASS
        order = {PASS: 0, WARN: 1, FAIL: 2}
        for tier in ("INVARIANT", "PHYSICAL", "REALISM"):
            rows = [r for r in self.rows if r[0] == tier]
            if not rows:
                continue
            print(f"\n=== {tier} ===")
            for _, status, name, detail in rows:
                if order[status] > order[worst]:
                    worst = status
                line = f"  {_MARK[status]} {name}"
                if detail:
                    line += f"  --  {detail}"
                print(line)
        n_fail = sum(1 for r in self.rows if r[1] == FAIL)
        n_warn = sum(1 for r in self.rows if r[1] == WARN)
        print(f"\n{'-'*64}\nSummary: {len(self.rows)} checks, "
              f"{n_fail} FAIL, {n_warn} WARN.")
        return worst


# --------------------------------------------------------------------------- #
#  Tier 1 -- invariants                                                        #
# --------------------------------------------------------------------------- #
def check_invariants(res, cfg, rep: Report):
    n_attacks_total = len(res.duration)

    rep.check("INVARIANT", "attack arrays same length",
              len(res.duration) == len(res.intensity) == len(res.patient_idx)
              == len(res.aborted),
              f"dur/int/idx/aborted = {len(res.duration)}/{len(res.intensity)}/"
              f"{len(res.patient_idx)}/{len(res.aborted)}")

    rep.check("INVARIANT", "sum(n_attacks) == #attacks",
              int(res.n_attacks.sum()) == n_attacks_total,
              f"sum(n_attacks)={int(res.n_attacks.sum()):,} vs rows={n_attacks_total:,}")

    rep.check("INVARIANT", "patient_idx in range",
              res.patient_idx.min() >= 0 and res.patient_idx.max() < cfg.n_patients)

    rep.check("INVARIANT", "intensity integer in 1..10",
              res.intensity.min() >= 1 and res.intensity.max() <= 10
              and np.issubdtype(res.intensity.dtype, np.integer),
              f"min={res.intensity.min()} max={res.intensity.max()} dtype={res.intensity.dtype}")

    # intrinsic (non-aborted) attacks obey the intrinsic floor; aborted attacks
    # can be shorter (an abortive can stop an attack below the intrinsic floor,
    # down to aborted_duration_floor_min). Both must respect the cap.
    intrinsic = res.duration[~res.aborted]
    ab = res.duration[res.aborted]
    rep.check("INVARIANT", "intrinsic duration within [floor, cap]",
              (intrinsic.size == 0 or
               (intrinsic.min() >= cfg.duration_floor_min - 1e-6
                and intrinsic.max() <= cfg.duration_cap_min + 1e-6)),
              f"min={intrinsic.min():.2f} max={intrinsic.max():.2f} "
              f"(floor {cfg.duration_floor_min}, cap {cfg.duration_cap_min})")
    rep.check("INVARIANT", "aborted duration >= aborted floor",
              ab.size == 0 or ab.min() >= cfg.aborted_duration_floor_min - 1e-6,
              f"min={ab.min():.2f} (aborted floor {cfg.aborted_duration_floor_min})")

    rep.check("INVARIANT", "n_attacks >= 1",
              res.n_attacks.min() >= 1, f"min={int(res.n_attacks.min())}")

    rep.check("INVARIANT", "preventives only reduce frequency",
              bool(np.all(res.n_attacks <= res.n_attacks_baseline)),
              f"max over-baseline = {int((res.n_attacks - res.n_attacks_baseline).max())}")

    rep.check("INVARIANT", "non-responders get no frequency cut",
              bool(np.all((res.prev_reduction > 0) <= res.on_preventive)),
              "every patient with reduction>0 is on a preventive")

    # determinism: same seed -> identical; different seed -> different
    again = simulate(cfg)
    rep.check("INVARIANT", "deterministic for a fixed seed",
              bool(np.array_equal(again.duration, res.duration)
                   and np.array_equal(again.intensity, res.intensity)))

    # global scaling consistency
    expected_scale = res.n_sufferers_global / cfg.n_patients
    rep.check("INVARIANT", "scale_factor = sufferers / n_patients",
              abs(res.scale_factor - expected_scale) < 1e-6,
              f"{res.scale_factor:.2f} vs {expected_scale:.2f}")

    # turning preventives off must zero the averted metric
    off = simulate(cfg, preventive_access_fraction=0.0)
    rep.check("INVARIANT", "access=0 averts nothing",
              off.summary()["global_attacks_averted_by_preventive"] == 0.0)


# --------------------------------------------------------------------------- #
#  Tier 2 -- physical plausibility (per-patient outlier scan)                  #
# --------------------------------------------------------------------------- #
def check_physical(res, cfg, rep: Report):
    n = len(res.n_attacks)
    # per-patient minutes spent in attack across the year
    pat_min = np.bincount(res.patient_idx, weights=res.duration, minlength=n)
    hours_per_year = pat_min / 60.0

    # (a) UNIVERSAL impossibility (denominator-free): a person cannot be in attack
    #     for more than the whole year (8760 h).
    over_year = int((hours_per_year > 8760).sum())
    rep.check("PHYSICAL", "no patient in attack > 8760 h/yr (whole year)",
              over_year == 0,
              f"{over_year} patient(s); max = {hours_per_year.max():,.0f} h/yr "
              f"({100*hours_per_year.max()/8760:.0f}% of the year)")

    # (b) clinically extraordinary: >50% of the entire year in attack -> WARN
    over_half = int((hours_per_year > 4380).sum())
    rep.check("PHYSICAL", "few patients in attack >50% of the year",
              over_half <= max(1, int(0.001 * n)),
              f"{over_half} patient(s) (~{100*over_half/n:.3f}%) over 4380 h/yr",
              warn_only=True)

    # Density checks need a meaningful active period. Episodic patients whose
    # sampled bout rounds to a sub-day "bout" (but are floored to >=1 attack)
    # have a near-zero denominator that makes per-day density meaningless -- they
    # are reported separately, not failed.
    meaningful = res.active_days >= 1.0
    tiny = int((~meaningful).sum())
    ad = res.active_days[meaningful]
    min_per_active_day = pat_min[meaningful] / ad
    attacks_per_active_day = res.n_attacks[meaningful] / ad

    # (c) physically impossible per active day: >24h/day in attack
    impossible_day = int((min_per_active_day > 1440 + 1e-6).sum())
    rep.check("PHYSICAL", "no patient in attack >24h/active-day",
              impossible_day == 0,
              f"{impossible_day} patient(s); worst "
              f"{min_per_active_day.max()/60:.1f} h/active-day "
              f"(over {meaningful.sum():,} patients with a >=1-day active period)")

    # (d) >16h/active-day -> WARN (rare, the freq x duration joint tail)
    extreme_day = int((min_per_active_day > 960).sum())
    rep.check("PHYSICAL", "few patients in attack >16h/active-day",
              extreme_day <= max(1, int(0.005 * meaningful.sum())),
              f"{extreme_day} patient(s) (~{100*extreme_day/meaningful.sum():.2f}%)",
              warn_only=True)

    # (e) ICHD ceiling on attack density (rounding tolerance 0.5; clipping should
    #     guarantee the underlying rate <= max_attacks_per_day).
    tol = cfg.max_attacks_per_day + 0.5
    over_ceiling = int((attacks_per_active_day > tol).sum())
    rep.check("PHYSICAL", "attacks/active-day <= ICHD ceiling (+rounding)",
              over_ceiling == 0,
              f"{over_ceiling} patient(s) over {tol}/day "
              f"(max {attacks_per_active_day.max():.2f})")

    # (f) nobody pinned to the safety cap (would mean the distribution runs too hot)
    at_cap = int((res.n_attacks >= cfg.max_attacks_per_patient).sum())
    rep.check("PHYSICAL", "no patient pinned to attacks/yr safety cap",
              at_cap == 0,
              f"{at_cap} patient(s) at cap {cfg.max_attacks_per_patient}; "
              f"max attacks/yr = {int(res.n_attacks.max()):,}")

    print(f"\n  info: {tiny} patient(s) (~{100*tiny/n:.2f}%) have a sub-1-day "
          f"sampled bout (floored to >=1 attack); excluded from density checks.")

    # (g) report the heaviest patients for eyeballing
    apad_all = res.n_attacks / np.maximum(res.active_days, 1e-9)
    mpad_all = pat_min / np.maximum(res.active_days, 1e-9)
    print("  heaviest patients (by attacks/yr):")
    top = np.argsort(res.n_attacks)[::-1][:5]
    print(f"    {'subtype':9s} {'attacks/yr':>10s} {'atk/active-day':>14s} "
          f"{'h/yr':>7s} {'h/active-day':>12s}")
    for i in top:
        sub = "episodic" if res.is_episodic[i] else "chronic"
        print(f"    {sub:9s} {int(res.n_attacks[i]):>10,d} "
              f"{apad_all[i]:>14.2f} {hours_per_year[i]:>7.0f} "
              f"{mpad_all[i]/60:>12.1f}")


# --------------------------------------------------------------------------- #
#  Tier 3 -- population realism vs literature                                  #
# --------------------------------------------------------------------------- #
def check_realism(res, cfg, rep: Report):
    s = res.summary()
    ep = res.is_episodic
    untreated_dur = res.duration[~res.aborted]

    rep.range("REALISM", "% episodic ~80%", s["pct_episodic"], 75.0, 90.0, "{:.1f}")
    rep.range("REALISM", "mean peak intensity ~7", s["attack_intensity_mean"],
              6.5, 7.5, "{:.2f}")
    rep.range("REALISM", "median attack duration 25-60 min",
              s["attack_duration_median_min"], 25.0, 60.0, "{:.1f}")

    within = float(np.mean((untreated_dur >= 15) & (untreated_dur <= 180)))
    rep.range("REALISM", "untreated attacks within ICHD 15-180 min (majority)",
              within, 0.60, 1.0, "{:.2f}")

    ep_mean = float(res.n_attacks[ep].mean())
    ch_mean = float(res.n_attacks[~ep].mean())
    rep.range("REALISM", "episodic mean attacks/yr ~100-260", ep_mean, 100, 260, "{:.0f}")
    rep.range("REALISM", "chronic mean attacks/yr ~550-950", ch_mean, 550, 950, "{:.0f}")
    rep.check("REALISM", "chronic > episodic frequency", ch_mean > ep_mean,
              f"chronic {ch_mean:.0f} vs episodic {ep_mean:.0f}", warn_only=True)

    rep.range("REALISM", "overall median attacks/yr ~40-150",
              s["attacks_per_year_median"], 40, 150, "{:.0f}")

    # heavy right tail: mean should exceed median for both freq and duration
    rep.check("REALISM", "attacks/yr right-skewed (mean>median)",
              s["attacks_per_year_mean"] > s["attacks_per_year_median"],
              f"mean {s['attacks_per_year_mean']:.0f} > median "
              f"{s['attacks_per_year_median']:.0f}", warn_only=True)
    rep.check("REALISM", "duration right-skewed (mean>median)",
              s["attack_duration_mean_min"] > s["attack_duration_median_min"],
              warn_only=True)

    rep.range("REALISM", "global sufferers ~2.5-3.5M (default prevalence)",
              s["n_sufferers_global"], 2.5e6, 3.5e6, "{:,.0f}")
    rep.range("REALISM", "preventive responders ~ access x responder-rate",
              s["pct_preventive_responders"],
              100 * cfg.preventive_access_fraction * cfg.preventive_responder_mean - 2,
              100 * cfg.preventive_access_fraction * cfg.preventive_responder_mean + 2,
              "{:.1f}")


# --------------------------------------------------------------------------- #
#  Eyeball sample                                                              #
# --------------------------------------------------------------------------- #
def show_examples(res, k=8, seed=0):
    rng = np.random.default_rng(seed)
    n = len(res.n_attacks)
    pick = rng.choice(n, size=min(k, n), replace=False)
    print("\n=== EYEBALL: random simulated patients ===")
    print(f"  {'#':>5s} {'subtype':9s} {'abort':5s} {'prev':5s} {'atk/yr':>7s} "
          f"{'med.dur':>8s} {'mean.int':>9s} {'h/yr':>6s}")
    for i in pick:
        m = res.patient_idx == i
        dur, inten = res.duration[m], res.intensity[m]
        print(f"  {i:>5d} {'episodic' if res.is_episodic[i] else 'chronic':9s} "
              f"{'yes' if res.has_access[i] else 'no':5s} "
              f"{'yes' if res.on_preventive[i] else 'no':5s} "
              f"{int(res.n_attacks[i]):>7d} {np.median(dur):>8.1f} "
              f"{inten.mean():>9.2f} {dur.sum()/60:>6.0f}")
    # a couple of full attack lists
    print("\n  example attacks (duration_min, peak) for the first two picked:")
    for i in pick[:2]:
        atks = res.patient_attacks(int(i))
        head = ", ".join(f"({d:.0f},{p})" for d, p in atks[:8])
        print(f"    patient {i}: {head}{' ...' if len(atks) > 8 else ''}")


# --------------------------------------------------------------------------- #
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cfg = Config(n_patients=n, seed=seed)
    res = simulate(cfg)

    print(f"Validating CH simulation: n_patients={n:,}, seed={seed}, "
          f"{len(res.duration):,} attacks simulated.")

    rep = Report()
    check_invariants(res, cfg, rep)
    check_physical(res, cfg, rep)
    check_realism(res, cfg, rep)
    show_examples(res)

    worst = rep.print()
    if worst == FAIL:
        print("\nRESULT: FAIL -- an invariant or physical check broke. See above.")
        sys.exit(1)
    elif worst == WARN:
        print("\nRESULT: PASS with warnings (realism drift is expected off-defaults).")
    else:
        print("\nRESULT: all checks passed.")


if __name__ == "__main__":
    main()
