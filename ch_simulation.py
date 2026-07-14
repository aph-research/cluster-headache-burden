"""
Monte Carlo simulation of cluster headache (CH) attacks.

Generates a synthetic population of CH patients and simulates every attack each
patient has in one year. Each attack is a tuple:

        (duration_minutes, max_intensity)      # max_intensity is an integer 1-10

Outputs (see SimulationResult):
  1. RAW per-patient data  -> .patient_attacks(i); flat arrays; .export_csv().
  2. GLOBAL intensity distribution -> .intensity_distribution() (by PEAK) and
     .time_at_levels() (true minutes spent AT each level via the within-attack
     profile), scaled to the worldwide CH population via annual prevalence.

Provenance for every default: CH_simulation_sources.md. Key modelling decisions:

  * Patients are stratified by subtype (episodic/chronic) and a *continuous*
    treatment-efficacy parameter (not a hard treated/untreated binary).
  * Intensity is a TWO-LEVEL model: each patient has a latent severity
    (between-patient SD), and their attacks vary around it (within-patient SD).
    Aggregated studies cannot separate these, so both are explicit levers.
    Snoer 2019 found LOW within-patient variability -> between dominates.
  * A treatment's robust effect is to ABORT the attack (truncate DURATION to the
    time-to-relief). The peak comes early (~9 min, before the abortive fully
    acts; Snoer treated 7.3 vs untreated 7.0), so any peak blunting is modest --
    `treated_peak_intensity_reduction` (default 0.1) shaves the peak of aborted
    attacks, e.g. from treating at onset. Set 0 to assume no peak effect.
  * Per-attack peak comes from PROSPECTIVE diaries (mean ~7), NOT the
    ceiling-loaded retrospective 9.7/72%-at-10 (Burish) -- that is a recalled
    "worst-ever" rating, not a per-attack measurement.
  * PREVENTIVES act on a separate channel from abortives: they mainly cut attack
    FREQUENCY (annual attack count), via shorter bouts (episodic) or lower daily
    frequency (chronic). Among responders they may also modestly blunt the PEAK
    of the attacks that remain (`preventive_peak_intensity_reduction`, default
    0.1), but never per-attack DURATION. Modelled as a responder rate + a
    fractional frequency cut (and optional peak cut) among responders.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields, asdict
import csv
import numpy as np


# --------------------------------------------------------------------------- #
#  Configuration -- every modelling assumption is a lever here.               #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # ---- Scope / global scaling ------------------------------------------- #
    annual_prevalence_per_100k: float = 53.0     # Fischera 2007 (1-yr prevalence, ADULTS)
    # 53/100k is an ADULT (18+) prevalence -> apply to the adult population, not 8.1B.
    # ~72% of 8.1B are 18+. 53/100k * 5.8e9 = ~3.07M, matching the ~3M EA-draft figure.
    adult_population: float = 5.8e9
    n_patients: int = 8_000                       # simulated cohort size (run-to-run
    # spread <~2% on common metrics, ~4% on the ≥9/10 tail -- negligible vs the
    # model's real error bars; 20k added reproducibility but no accuracy)
    seed: int = 0

    # ---- Subtype split ---------------------------------------------------- #
    episodic_fraction: float = 0.85               # Schindler&Burish; Fischera 6:1; ~80-90% episodic

    # ---- Treatment: access (patient-level) -------------------------------- #
    # Global fraction with real access to an effective abortive (O2 / SC-nasal
    # triptan). Independent estimate ~0.18 (range 0.10-0.30): HIC ~0.55, MIC ~0.12,
    # LIC ~0.03, pop-weighted (~85% of world is LMIC). Only the HIC input is
    # measured (Rossi 2020 EU 47% unrestricted); LMIC inputs are inferences bounded
    # by WHO neurological treatment-gap data. First parameter to vary in sensitivity.
    treatment_access_fraction: float = 0.18

    # ---- Treatment: per-patient abortive efficacy ------------------------- #
    abort_prob_mean: float = 0.64                 # O2+triptan participant-weighted (Rusanen 2022)
    abort_prob_sd: float = 0.22
    treat_fraction: float = 0.85                  # share of attacks actually treated (Snoer)

    # ---- Treatment: effect magnitudes ------------------------------------- #
    aborted_duration_mean_min: float = 15.0       # time-to-relief (suma ~7-10, O2 by 15)
    aborted_duration_sd_min: float = 6.0
    aborted_duration_floor_min: float = 3.0
    treated_peak_intensity_reduction: float = 0.1 # peak-pain cut on aborted attacks (fraction in [0,1])

    # ---- Preventive treatment: reduces attack FREQUENCY (+ optional peak) -- #
    # A preventive (verapamil first-line; corticosteroids transitional; lithium,
    # topiramate, melatonin, galcanezumab) lowers HOW MANY attacks a patient has
    # -- via shorter bouts (episodic) or lower daily frequency (chronic). Modelled
    # as a fractional cut to the patient's ANNUAL attack count, independent of
    # abortive access. Among responders it may ALSO modestly blunt the PEAK of the
    # attacks that remain (preventive_peak_intensity_reduction); it never changes
    # per-attack DURATION.
    preventive_access_fraction: float = 0.23      # bottom-up HIC/MIC/LIC (see sources)
    preventive_responder_mean: float = 0.42       # P(responds | on preventive); Rusanen 2022
    preventive_responder_reduction_mean: float = 0.55  # freq cut among responders
    preventive_responder_reduction_sd: float = 0.20
    preventive_reduction_floor: float = 0.50      # responder == >=50% reduction (trial convention)
    preventive_reduction_cap: float = 0.95
    preventive_peak_intensity_reduction: float = 0.1  # peak-pain cut on responders' remaining attacks

    # ---- Per-attack PEAK INTENSITY: two-level model (NRS 1..10) ----------- #
    intensity_mean: float = 7.0                   # population mean peak (Snoer ~7.0)
    intensity_between_sd: float = 1.2             # between-patient severity SD
    intensity_within_sd: float = 0.9             # within-patient attack-to-attack SD
    # (marginal SD ~ sqrt(1.6^2+1.0^2) = 1.89, matching Snoer ~2.3 / Russell spread)

    # ---- Per-attack DURATION (intrinsic/untreated), minutes --------------- #
    duration_median_min: float = 45.0             # prospective ~39 (Hagedorn); skewed
    duration_sigma: float = 0.70                  # lognormal shape -> tail to ~180+
    duration_floor_min: float = 5.0
    duration_cap_min: float = 360.0
    # Intensity<->duration coupling, CENTERED on the population mean so the median
    # duration stays calibrated: factor = 1 + slope*(intensity - intensity_mean),
    # clipped. slope=0 -> no coupling. Russell 1981 found a positive correlation;
    # Hagedorn implies slope ~0.1 but n=1 over a narrow range, so default is gentler.
    dur_intensity_slope: float = 0.08
    dur_intensity_factor_min: float = 0.3
    dur_intensity_factor_max: float = 2.5

    # ---- Within-attack intensity profile ---------------------------------- #
    # DEFAULT model: FIXED-MINUTES rise & decline (from primary data: Torelli
    # time-to-peak ~9 min; Snoer resolution ~20 min); plateau = remaining duration
    # (so short/aborted attacks have little/no plateau). This is independent of
    # any prior burden model.
    profile_rise_min: float = 9.0
    profile_decline_min: float = 20.0
    # COMPARISON model (fractional 15/70/15) -- kept only for the side-by-side
    # chart; NOT the default. (This 70%-plateau split comes from a prior published
    # burden model, so it is shown for contrast, not used for headline numbers.)
    profile_rise_frac: float = 0.15
    profile_plateau_frac: float = 0.70
    profile_decline_frac: float = 0.15

    # ---- FREQUENCY: shared bounds (ICHD criterion D) ---------------------- #
    max_attacks_per_day: float = 8.0              # ICHD ceiling (>8/day => reconsider dx)
    min_attacks_per_day: float = 0.5              # ICHD floor (1 every other day)
    episodic_max_active_weeks: float = 39.0       # episodic needs >=3mo (13wk) remission

    # ---- FREQUENCY: episodic ---------------------------------------------- #
    e_bouts_per_year_mean: float = 1.2            # Gaul 2012
    e_bouts_per_year_sd: float = 1.1
    e_bout_weeks_mean: float = 8.5                # Gaul 2012
    e_bout_weeks_sd: float = 5.7
    e_attacks_per_day_mean: float = 2.0           # episodic median ~1-2 (Vikelis); 1.1-3.1
    e_attacks_per_day_sd: float = 1.5

    # ---- FREQUENCY: chronic ----------------------------------------------- #
    c_active_fraction: float = 0.90               # remissions <3mo by definition
    c_attacks_per_day_mean: float = 2.5           # chronic median ~2 (Vikelis); Gaul 3.3
    c_attacks_per_day_sd: float = 1.8

    max_attacks_per_patient: int = 3000           # loose safety net (~8/day * 365)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _gamma(rng, mean, sd, size):
    """Gamma draws parameterised by mean and sd (positive, right-skewed)."""
    mean = max(mean, 1e-9)
    sd = max(sd, 1e-9)
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    return rng.gamma(shape, scale, size)


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    cfg: Config
    is_episodic: np.ndarray            # per-patient
    has_access: np.ndarray
    efficacy: np.ndarray
    patient_severity: np.ndarray       # latent per-patient mean peak intensity
    n_attacks: np.ndarray
    n_attacks_baseline: np.ndarray     # per-patient attacks BEFORE preventive reduction
    active_days: np.ndarray            # per-patient days/yr in an active period
    on_preventive: np.ndarray          # per-patient
    prev_reduction: np.ndarray         # per-patient fractional frequency reduction
    patient_idx: np.ndarray            # per-attack
    duration: np.ndarray               # minutes
    intensity: np.ndarray              # integer 1..10 (peak)
    aborted: np.ndarray                # bool
    scale_factor: float
    n_sufferers_global: float

    # -- raw access ---------------------------------------------------------- #
    def patient_attacks(self, i: int):
        """List of (duration_min, max_intensity) tuples for simulated patient i."""
        m = self.patient_idx == i
        return list(zip(self.duration[m].round(2).tolist(), self.intensity[m].tolist()))

    def grouped_csv(self) -> str:
        """One row per patient as a proper RFC-4180 CSV (comma-delimited). Columns:
        type, treatment, total_attacks, total_duration, attacks
        where `attacks` is ";"-joined (duration_int,intensity.2f) tuples. The
        `attacks` field contains commas, so it is automatically double-quoted ->
        opens correctly in Google Sheets / Excel without a custom delimiter.
        (`treatment` = patient has treatment access.)"""
        import io
        buf = io.StringIO()
        w = csv.writer(buf)  # default: comma delimiter, QUOTE_MINIMAL
        w.writerow(["type", "treatment", "total_attacks", "total_duration", "attacks"])
        offs = np.concatenate(([0], np.cumsum(self.n_attacks)))
        dur, inten = self.duration, self.intensity
        ep, acc = self.is_episodic, self.has_access
        for i in range(len(ep)):
            a, b = int(offs[i]), int(offs[i + 1])
            d, it = dur[a:b], inten[a:b]
            tuples = ";".join(f"({int(round(float(x)))},{float(y):.2f})"
                              for x, y in zip(d, it))
            w.writerow([
                "episodic" if ep[i] else "chronic",
                "TRUE" if acc[i] else "FALSE",
                int(self.n_attacks[i]),
                int(round(float(d.sum()))),
                tuples,
            ])
        return buf.getvalue()

    def export_csv(self, path: str, max_rows: int | None = None):
        """Write raw per-attack rows: patient_id, subtype, has_access, efficacy,
        duration_min, peak_intensity, aborted."""
        n = len(self.duration) if max_rows is None else min(max_rows, len(self.duration))
        ep, acc, eff = self.is_episodic, self.has_access, self.efficacy
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "subtype", "has_access", "efficacy",
                        "duration_min", "peak_intensity", "aborted"])
            for j in range(n):
                p = self.patient_idx[j]
                w.writerow([int(p), "episodic" if ep[p] else "chronic",
                            int(acc[p]), round(float(eff[p]), 3),
                            round(float(self.duration[j]), 2),
                            int(self.intensity[j]), int(self.aborted[j])])
        return path

    # -- aggregate outputs --------------------------------------------------- #
    def intensity_distribution(self, weight: str = "attacks", scaled: bool = True):
        """Global distribution over PEAK intensity levels 1..10.
        weight='attacks' -> attack counts; weight='minutes' -> attack-minutes by peak.
        (For minutes ACTUALLY spent at each level, use time_at_levels().)"""
        levels = np.arange(1, 11)
        if weight == "attacks":
            vals = np.array([(self.intensity == k).sum() for k in levels], float)
        elif weight == "minutes":
            vals = np.array([self.duration[self.intensity == k].sum() for k in levels], float)
        else:
            raise ValueError("weight must be 'attacks' or 'minutes'")
        return vals * self.scale_factor if scaled else vals

    def time_at_levels(self, scaled: bool = True, mode: str = "minutes"):
        """Global minutes ACTUALLY spent at each intensity level 1..10, using the
        within-attack profile. Linear rise/decline ramps spread time uniformly
        across levels 1..peak; the plateau sits at the peak.

        mode="minutes"   -> DEFAULT. Fixed-minutes rise/decline (Torelli/Snoer);
                            plateau = duration - rise - decline (>=0). Short attacks
                            get little/no plateau.
        mode="fractions" -> COMPARISON ONLY. The 15/70/15 fractional split from a
                            prior published burden model. For side-by-side contrast.
        """
        c = self.cfg
        inten, D = self.intensity, self.duration
        if mode == "minutes":
            nominal = c.profile_rise_min + c.profile_decline_min
            ramp = np.minimum(nominal, D)
            plateau = D - ramp
        elif mode == "fractions":
            tot = c.profile_rise_frac + c.profile_plateau_frac + c.profile_decline_frac
            plateau = (c.profile_plateau_frac / tot) * D
            ramp = ((c.profile_rise_frac + c.profile_decline_frac) / tot) * D
        else:
            raise ValueError("mode must be 'minutes' or 'fractions'")
        out = np.zeros(10)
        for k in range(1, 11):
            out[k - 1] += plateau[inten == k].sum()
            ge = inten >= k
            out[k - 1] += (ramp[ge] / inten[ge]).sum()
        return out * self.scale_factor if scaled else out

    def attacks_per_year_percentiles(self, qs=(10, 25, 50, 75, 90, 95, 99)):
        return {q: float(np.percentile(self.n_attacks, q)) for q in qs}

    def summary(self) -> dict:
        cfg = self.cfg
        py = lambda mins: mins * self.scale_factor / 60 / 24 / 365
        tal = self.time_at_levels(scaled=False)
        return {
            "n_patients_simulated": cfg.n_patients,
            "n_sufferers_global": self.n_sufferers_global,
            "pct_episodic": 100 * self.is_episodic.mean(),
            "pct_with_access": 100 * self.has_access.mean(),
            "pct_on_preventive": 100 * self.on_preventive.mean(),
            "pct_preventive_responders": 100 * float((self.prev_reduction > 0).mean()),
            "global_attacks_averted_by_preventive": float(
                (self.n_attacks_baseline.sum() - self.n_attacks.sum()) * self.scale_factor),
            "pct_attacks_averted_by_preventive": 100 * (
                1.0 - self.n_attacks.sum() / max(1, self.n_attacks_baseline.sum())),
            "attacks_per_year_mean": float(self.n_attacks.mean()),
            "attacks_per_year_median": float(np.median(self.n_attacks)),
            "attacks_per_year_mean_episodic": float(self.n_attacks[self.is_episodic].mean()),
            "attacks_per_year_mean_chronic": float(self.n_attacks[~self.is_episodic].mean()),
            "attack_duration_mean_min": float(self.duration.mean()),
            "attack_duration_median_min": float(np.median(self.duration)),
            "attack_intensity_mean": float(self.intensity.mean()),
            "pct_attacks_aborted": 100 * float(self.aborted.mean()),
            "global_attacks_per_year": float(self.intensity_distribution("attacks").sum()),
            "global_person_years_in_attack": py(self.duration.sum()),
            "global_person_days_in_attack": py(self.duration.sum()) * 365.0,
            # person-years AT each level (true time-at-level, not by-peak):
            "global_person_years_at_ge7": py(tal[6:].sum()),
            "global_person_years_at_ge9": py(tal[8:].sum()),
            "global_person_years_at_10": py(tal[9]),
            # same quantities in person-DAYS (handy for the rarest, most-severe time)
            "global_person_days_at_ge9": py(tal[8:].sum()) * 365.0,
            "global_person_days_at_10": py(tal[9]) * 365.0,
        }


# --------------------------------------------------------------------------- #
#  Core simulation                                                            #
# --------------------------------------------------------------------------- #
def simulate(cfg: Config | None = None, **overrides) -> SimulationResult:
    """Run the simulation. Pass a Config and/or keyword lever overrides."""
    cfg = cfg or Config()
    if overrides:
        cfg = replace(cfg, **overrides)
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_patients

    # ---- per-patient attributes ------------------------------------------ #
    is_episodic = rng.random(n) < cfg.episodic_fraction
    has_access = rng.random(n) < cfg.treatment_access_fraction
    efficacy = np.where(
        has_access,
        np.clip(rng.normal(cfg.abort_prob_mean, cfg.abort_prob_sd, n), 0.0, 0.95),
        0.0,
    )
    # latent per-patient severity (between-patient variation)
    patient_severity = np.clip(
        rng.normal(cfg.intensity_mean, cfg.intensity_between_sd, n), 1.0, 10.0)

    # ---- attacks per year per patient ------------------------------------ #
    apd_lo, apd_hi = cfg.min_attacks_per_day, cfg.max_attacks_per_day
    bouts = _gamma(rng, cfg.e_bouts_per_year_mean, cfg.e_bouts_per_year_sd, n)
    bout_weeks = _gamma(rng, cfg.e_bout_weeks_mean, cfg.e_bout_weeks_sd, n)
    e_apd = np.clip(_gamma(rng, cfg.e_attacks_per_day_mean, cfg.e_attacks_per_day_sd, n),
                    apd_lo, apd_hi)
    # episodic: active time bounded so remission stays >= 3 months (39-week cap)
    active_weeks = np.minimum(bouts * bout_weeks, cfg.episodic_max_active_weeks)
    epis = active_weeks * 7.0 * e_apd

    c_apd = np.clip(_gamma(rng, cfg.c_attacks_per_day_mean, cfg.c_attacks_per_day_sd, n),
                    apd_lo, apd_hi)
    chron = cfg.c_active_fraction * 365.0 * c_apd

    # days/yr a patient is in an active period (bouts for episodic; ~year-round for
    # chronic). Used to sanity-check attack density (e.g. minutes-in-attack/active-day).
    active_days = np.where(is_episodic, active_weeks * 7.0,
                           cfg.c_active_fraction * 365.0)

    n_attacks_base = np.where(is_episodic, epis, chron)
    n_attacks_baseline = np.clip(np.round(n_attacks_base), 1,
                                 cfg.max_attacks_per_patient).astype(int)

    # ---- preventive treatment: cut annual attack frequency --------------- #
    # Patient is on a preventive (access) -> may be a responder -> annual attacks
    # are cut by a fractional reduction. Non-responders get no frequency benefit.
    on_preventive = rng.random(n) < cfg.preventive_access_fraction
    prev_responder = on_preventive & (rng.random(n) < cfg.preventive_responder_mean)
    prev_reduction = np.where(
        prev_responder,
        np.clip(rng.normal(cfg.preventive_responder_reduction_mean,
                           cfg.preventive_responder_reduction_sd, n),
                cfg.preventive_reduction_floor, cfg.preventive_reduction_cap),
        0.0,
    )
    n_attacks = np.clip(np.round(n_attacks_base * (1.0 - prev_reduction)), 1,
                        cfg.max_attacks_per_patient).astype(int)

    total = int(n_attacks.sum())
    patient_idx = np.repeat(np.arange(n), n_attacks)

    # ---- per-attack peak intensity (two-level: patient mean + within noise) #
    latent = patient_severity[patient_idx] + rng.normal(0, cfg.intensity_within_sd, total)
    intensity = np.clip(np.round(latent), 1, 10).astype(int)

    # ---- intrinsic duration (coupled to intensity) ----------------------- #
    base = cfg.duration_median_min * np.exp(cfg.duration_sigma * rng.standard_normal(total))
    factor = np.clip(1.0 + cfg.dur_intensity_slope * (intensity - cfg.intensity_mean),
                     cfg.dur_intensity_factor_min, cfg.dur_intensity_factor_max)
    duration = np.clip(base * factor, cfg.duration_floor_min, cfg.duration_cap_min)

    # ---- treatment: abort some attacks ----------------------------------- #
    # Only an effective abortive truncates an attack. Naturally short attacks are
    # NOT modelled here -- they already fall out of the intrinsic (untreated)
    # duration distribution above, which is calibrated to the literature.
    treated = has_access[patient_idx] & (rng.random(total) < cfg.treat_fraction)
    aborted = treated & (rng.random(total) < efficacy[patient_idx])

    abort_dur = np.clip(
        rng.normal(cfg.aborted_duration_mean_min, cfg.aborted_duration_sd_min, total),
        cfg.aborted_duration_floor_min, None)
    duration = np.where(aborted, np.minimum(duration, abort_dur), duration)

    # ---- treatment: peak-intensity reduction ----------------------------- #
    # Applied AFTER duration so it doesn't feed the intensity->duration coupling.
    # Preventive blunts the peak of a responder's remaining attacks; an abortive
    # blunts the attacks it aborts; on an aborted attack of a responder both stack.
    if cfg.preventive_peak_intensity_reduction > 0:
        redp = np.clip(np.round(intensity * (1 - cfg.preventive_peak_intensity_reduction)), 1, 10)
        intensity = np.where(prev_responder[patient_idx], redp, intensity).astype(int)
    if cfg.treated_peak_intensity_reduction > 0:
        red = np.clip(np.round(intensity * (1 - cfg.treated_peak_intensity_reduction)), 1, 10)
        intensity = np.where(aborted, red, intensity).astype(int)

    n_glob = cfg.annual_prevalence_per_100k / 1e5 * cfg.adult_population
    return SimulationResult(
        cfg=cfg, is_episodic=is_episodic, has_access=has_access, efficacy=efficacy,
        patient_severity=patient_severity, n_attacks=n_attacks,
        n_attacks_baseline=n_attacks_baseline, active_days=active_days,
        on_preventive=on_preventive,
        prev_reduction=prev_reduction, patient_idx=patient_idx,
        duration=duration, intensity=intensity, aborted=aborted,
        scale_factor=n_glob / n, n_sufferers_global=n_glob,
    )


# --------------------------------------------------------------------------- #
#  Lever sweep / sensitivity                                                  #
# --------------------------------------------------------------------------- #
def sweep(param: str, values, metric: str, base: Config | None = None):
    """Vary one Config lever over `values`; return [(value, metric), ...].
    `metric` is any key from SimulationResult.summary()."""
    valid = {f.name for f in fields(Config)}
    if param not in valid:
        raise ValueError(f"unknown lever '{param}'")
    out = []
    for v in values:
        res = simulate(base, **{param: v})
        out.append((v, res.summary()[metric]))
    return out


# --------------------------------------------------------------------------- #
#  Counterfactual access (for cost-effectiveness analysis)                    #
# --------------------------------------------------------------------------- #
CF_CHANNELS = ("abortive", "preventive", "both")


def _counterfactual(cfg: Config, want_tuples: bool = False) -> dict:
    """Core counterfactual: per-(currently-untreated)-patient baseline burden and
    the burden AVERTED by gaining access, under three interventions (abortive /
    preventive / both). Captures the real shape of the benefit -- abortives
    truncate DURATION (same attack count), preventives remove whole ATTACKS, and
    each optionally blunts the PEAK of the attacks it acts on. Severity time
    (DLES = person-days at >=9) is scored with the SAME within-attack profile as
    the burden view (time_at_levels), not the whole duration of peak->=9 attacks.

    Returns numpy arrays (length n_patients): is_episodic, random_id, base_attacks,
    base_min, base_severe, base_dles, and `{channel}_{attacks,min,severe,dles}_averted`.
    If want_tuples, also returns `_tuples` (per-patient attack-tuple strings)."""
    base = simulate(replace(cfg, treatment_access_fraction=0.0,
                            preventive_access_fraction=0.0))
    n = cfg.n_patients
    rp = np.random.default_rng(cfg.seed + 101)   # preventive draws
    ra = np.random.default_rng(cfg.seed + 202)   # abortive draws

    responder = rp.random(n) < cfg.preventive_responder_mean
    reduction = np.where(
        responder,
        np.clip(rp.normal(cfg.preventive_responder_reduction_mean,
                          cfg.preventive_responder_reduction_sd, n),
                cfg.preventive_reduction_floor, cfg.preventive_reduction_cap),
        0.0)
    eff = np.clip(ra.normal(cfg.abort_prob_mean, cfg.abort_prob_sd, n), 0.0, 0.95)
    rid = np.random.default_rng(cfg.seed + 303).random(n)   # for sampling

    offs = np.concatenate(([0], np.cumsum(base.n_attacks)))
    dur, inten = base.duration, base.intensity

    out = {
        "is_episodic": base.is_episodic, "random_id": rid,
        "base_attacks": base.n_attacks.astype(float),
        "base_min": np.zeros(n), "base_severe": np.zeros(n), "base_dles": np.zeros(n),
    }
    for c in CF_CHANNELS:
        for m in ("attacks_averted", "min_averted", "severe_averted", "dles_averted"):
            out[f"{c}_{m}"] = np.zeros(n)
    tuples = {k: [None] * n for k in ("base", *CF_CHANNELS)} if want_tuples else None

    def tup(d, it):
        return ";".join(f"({int(round(float(x)))},{float(y):.2f})"
                        for x, y in zip(d, it))

    # minutes an attack actually spends at intensity >=9, via the SAME within-attack
    # profile as SimulationResult.time_at_levels (plateau at the peak + the ramp's
    # share of levels 9..peak) -- so CE "DLES" matches the burden "DLES" definition.
    nominal = cfg.profile_rise_min + cfg.profile_decline_min

    def min_at_ge9(peak, D):
        ramp = np.minimum(nominal, D)
        plateau = D - ramp
        nlev = (peak >= 9).astype(float) + (peak >= 10).astype(float)  # 1 at peak 9, 2 at 10
        return np.where(peak >= 9, plateau + ramp / np.maximum(peak, 1) * nlev, 0.0)

    def pcut(peak, frac):   # fractional cut to the 1..10 peak (no-op when frac<=0)
        return np.clip(np.round(peak * (1 - frac)), 1, 10) if frac > 0 else peak

    for i in range(n):
        a, b = int(offs[i]), int(offs[i + 1])
        d = dur[a:b].astype(float)
        it = inten[a:b]
        nb = len(d)
        base_min = float(d.sum())
        sev_base = int((it >= 9).sum())
        dles_base = float(min_at_ge9(it, d).sum()) / 1440.0
        out["base_min"][i] = base_min
        out["base_severe"][i] = sev_base
        out["base_dles"][i] = dles_base

        # abortive: truncate aborted attacks' duration (per-attack on baseline)
        treated = ra.random(nb) < cfg.treat_fraction
        aborted = treated & (ra.random(nb) < eff[i])
        abort_dur = np.clip(ra.normal(cfg.aborted_duration_mean_min,
                                      cfg.aborted_duration_sd_min, nb),
                            cfg.aborted_duration_floor_min, None)
        d_trunc = np.where(aborted, np.minimum(d, abort_dur), d)

        # preventive: remove a random subset of whole attacks
        keep = np.ones(nb, dtype=bool)
        if reduction[i] > 0 and nb > 0:
            n_rm = int(round(reduction[i] * nb))
            if n_rm > 0:
                keep[rp.choice(nb, size=min(n_rm, nb), replace=False)] = False

        # peak-intensity blunting: preventive on a responder's remaining attacks,
        # abortive on the attacks it aborts; on the "both" channel they stack.
        it_pv = pcut(it, cfg.preventive_peak_intensity_reduction) if reduction[i] > 0 else it
        it_ab = np.where(aborted, pcut(it, cfg.treated_peak_intensity_reduction), it)
        it_bo = np.where(aborted, pcut(it_pv, cfg.treated_peak_intensity_reduction), it_pv)

        scen = {"abortive": (d_trunc, it_ab),
                "preventive": (d[keep], it_pv[keep]),
                "both": (d_trunc[keep], it_bo[keep])}
        for c, (du, iu) in scen.items():
            out[f"{c}_attacks_averted"][i] = nb - len(du)
            out[f"{c}_min_averted"][i] = base_min - float(du.sum())
            out[f"{c}_severe_averted"][i] = sev_base - int((iu >= 9).sum())
            out[f"{c}_dles_averted"][i] = dles_base - float(min_at_ge9(iu, du).sum()) / 1440.0
        if want_tuples:
            tuples["base"][i] = tup(d, it)
            for c, (du, iu) in scen.items():
                tuples[c][i] = tup(du, iu)
    if want_tuples:
        out["_tuples"] = tuples
    return out


def counterfactual_csv(cfg: Config | None = None, **overrides) -> str:
    """CSV of per-untreated-patient baseline + counterfactual 'with access' attacks
    (abortive / preventive / both), with the attack-tuple lists (same format as
    grouped_csv, so a spreadsheet's own DLES/severity formulas apply) plus averted
    columns (attacks / minutes / severe>=9 / DLES). See _counterfactual()."""
    import io
    cfg = cfg or Config()
    if overrides:
        cfg = replace(cfg, **overrides)
    d = _counterfactual(cfg, want_tuples=True)
    t = d["_tuples"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "id", "type", "random_id", "total_attacks", "total_duration",
        "severe_attacks_baseline", "dles_baseline", "attacks",
        "attacks_with_access_abortive", "attacks_averted_abortive",
        "time_averted_abortive_min", "severe_averted_abortive", "dles_averted_abortive",
        "attacks_with_access_preventive", "attacks_averted_preventive",
        "time_averted_preventive_min", "severe_averted_preventive", "dles_averted_preventive",
        "attacks_with_access_both", "attacks_averted_both",
        "time_averted_both_min", "severe_averted_both", "dles_averted_both",
    ])
    for i in range(cfg.n_patients):
        row = [i, "episodic" if d["is_episodic"][i] else "chronic",
               round(float(d["random_id"][i]), 9), int(d["base_attacks"][i]),
               int(round(d["base_min"][i])), int(d["base_severe"][i]),
               round(d["base_dles"][i], 5), t["base"][i]]
        for c in CF_CHANNELS:
            row += [t[c][i], int(d[f"{c}_attacks_averted"][i]),
                    int(round(d[f"{c}_min_averted"][i])),
                    int(d[f"{c}_severe_averted"][i]),
                    round(d[f"{c}_dles_averted"][i], 5)]
        w.writerow(row)
    return buf.getvalue()


def cost_effectiveness(cfg: Config | None = None, *, annual_budget: float = 100_000.0,
                       patients_reached: float = 500.0, channel: str = "both",
                       effect_size_mean: float = 0.6,
                       beneficiary_episodic_share: float | None = None,
                       **overrides) -> dict:
    """End-to-end cost-effectiveness of helping CH patients reach treatment.

    Model: each beneficiary captures, on average, `effect_size_mean` (= clinical
    capture) of the full no-treatment -> full-access benefit for the chosen
    `channel`. `patients_reached` is a COUNTERFACTUAL adopter count (people who got
    treated because of you and would not have otherwise); the "would-have-anyway"
    discount therefore lives in `patients_reached`, NOT in `effect_size_mean`.
    `beneficiary_episodic_share` lets you model targeting (None = population mix).

    NB: the per-patient `full_benefit` is measured against a fully-UNTREATED
    baseline, so `patients_reached` must be currently-untreated patients. Pushing
    reached toward the whole population with effect_size_mean ~1 measures against
    the untreated-world burden, which exceeds today's partly-treated burden.

        total_averted = patients_reached * effect_size_mean * mean(full_benefit)
        cost-effectiveness = annual_budget / total_averted
    """
    cfg = cfg or Config()
    if overrides:
        cfg = replace(cfg, **overrides)
    if channel not in CF_CHANNELS:
        raise ValueError(f"channel must be one of {CF_CHANNELS}")

    d = _counterfactual(cfg)
    ep = d["is_episodic"]

    def per_patient(metric):
        arr = d[f"{channel}_{metric}"]
        if beneficiary_episodic_share is None:
            return float(arr.mean())
        s = beneficiary_episodic_share
        me = float(arr[ep].mean()) if ep.any() else 0.0
        mc = float(arr[~ep].mean()) if (~ep).any() else 0.0
        return s * me + (1.0 - s) * mc

    f = effect_size_mean * patients_reached
    attacks = per_patient("attacks_averted") * f
    severe = per_patient("severe_averted") * f
    hours = per_patient("min_averted") * f / 60.0
    dles = per_patient("dles_averted") * f

    def ratio(x):
        return (annual_budget / x) if x > 0 else None

    return {
        "annual_budget": annual_budget, "patients_reached": patients_reached,
        "cost_per_patient": (annual_budget / patients_reached
                             if patients_reached else None),
        "channel": channel, "effect_size_mean": effect_size_mean,
        "beneficiary_episodic_share": beneficiary_episodic_share,
        "attacks_averted": attacks, "severe_attacks_averted": severe,
        "attack_hours_averted": hours, "dles_averted": dles,
        "cost_per_attack": ratio(attacks),
        "cost_per_severe_attack": ratio(severe),
        "cost_per_attack_hour": ratio(hours),
        "cost_per_dles": ratio(dles),
        "dles_per_1000usd": (dles / annual_budget * 1000.0) if annual_budget else None,
    }


# --------------------------------------------------------------------------- #
#  Monte-Carlo band helpers (uncertainty propagation)                         #
# --------------------------------------------------------------------------- #
# Cache of per-patient benefit (four floats) keyed on the simulation config. The
# cost-effectiveness sections depend on the sim only through this, and NOT on the
# budget / patients-reached / effect-size / funnel inputs -- so when only those
# text fields change, we skip the (slow, pure-Python) counterfactual loop and reuse
# the cached benefit. Also dedupes the ClusterFree + ClusterInfo panels, which
# otherwise recompute the identical counterfactual on every run.
_PP_CACHE = {}


def _per_patient_benefit(cfg, channel, beneficiary_episodic_share):
    """Mean per-(untreated)-patient averted {attacks,severe,min,dles} for `channel`."""
    key = (tuple(sorted(asdict(cfg).items())), channel, beneficiary_episodic_share)
    hit = _PP_CACHE.get(key)
    if hit is not None:
        return hit
    result = _compute_per_patient_benefit(cfg, channel, beneficiary_episodic_share)
    if len(_PP_CACHE) > 64:   # bound memory; sensitivity sweeps many distinct configs
        _PP_CACHE.clear()
    _PP_CACHE[key] = result
    return result


def _compute_per_patient_benefit(cfg, channel, beneficiary_episodic_share):
    d = _counterfactual(cfg)
    ep = d["is_episodic"]

    def pp(metric):
        arr = d[f"{channel}_{metric}"]
        if beneficiary_episodic_share is None:
            return float(arr.mean())
        s = beneficiary_episodic_share
        me = float(arr[ep].mean()) if ep.any() else 0.0
        mc = float(arr[~ep].mean()) if (~ep).any() else 0.0
        return s * me + (1.0 - s) * mc

    return {m: pp(m) for m in
            ("attacks_averted", "severe_averted", "min_averted", "dles_averted")}


def _trunc_normal(rng, mean, sd, n, lo=0.0, hi=1.0):
    """n draws from Normal(mean, sd) clipped to [lo, hi] (a simple truncated normal)."""
    return np.clip(rng.normal(float(mean), max(float(sd), 0.0), n), lo, hi)


def _bands(x):
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90),
            "mean": float(np.mean(x))}


def _metric_bands(pp, f, annual_budget):
    """Given per-patient benefit `pp` and a per-draw multiplier array `f`, build
    output+cost bands for all four metrics. Cost percentiles are flipped so p10 is
    pessimistic (expensive) and p90 optimistic (cheap)."""
    metrics = {
        "attacks_averted": pp["attacks_averted"] * f,
        "severe_attacks_averted": pp["severe_averted"] * f,
        "attack_hours_averted": pp["min_averted"] * f / 60.0,
        "dles_averted": pp["dles_averted"] * f,
    }
    ckey = {"attacks_averted": "cost_per_attack",
            "severe_attacks_averted": "cost_per_severe_attack",
            "attack_hours_averted": "cost_per_attack_hour",
            "dles_averted": "cost_per_dles"}
    out = {}
    for k, arr in metrics.items():
        out[k] = _bands(arr)
        c = np.where(arr > 0, annual_budget / np.where(arr > 0, arr, 1.0), np.nan)
        # higher averted -> lower cost, so the 90th pct of cost is the pessimistic p10
        p90c, p50c, p10c = np.nanpercentile(c, [10, 50, 90])
        out[ckey[k]] = {"p10": float(p10c), "p50": float(p50c), "p90": float(p90c),
                        "mean": float(np.nanmean(c))}
    return out


def cost_effectiveness_bands(cfg: Config | None = None, *, annual_budget: float = 100_000.0,
                             patients_reached: float = 500.0,
                             effect_size_mean: float = 0.6, effect_size_sd: float = 0.15,
                             channel: str = "both",
                             beneficiary_episodic_share: float | None = None,
                             n_mc: int = 5_000, **overrides) -> dict:
    """Like cost_effectiveness(), but the effect size is a truncated normal (median
    `effect_size_mean`, spread `effect_size_sd`, clipped to [0,1]) instead of a
    point value. `patients_reached` stays a point input. Propagates the effect-size
    uncertainty by Monte Carlo and returns p10/p50/p90 bands for every output."""
    cfg = cfg or Config()
    if overrides:
        cfg = replace(cfg, **overrides)
    if channel not in CF_CHANNELS:
        raise ValueError(f"channel must be one of {CF_CHANNELS}")
    pp = _per_patient_benefit(cfg, channel, beneficiary_episodic_share)
    rng = np.random.default_rng(cfg.seed + 505)
    eff = _trunc_normal(rng, effect_size_mean, effect_size_sd, n_mc)
    out = _metric_bands(pp, patients_reached * eff, annual_budget)
    out.update({
        "annual_budget": annual_budget, "patients_reached": patients_reached,
        "channel": channel, "n_mc": n_mc,
        "beneficiary_episodic_share": beneficiary_episodic_share,
        "effect_size_mean": _bands(eff),
    })
    return out


# --------------------------------------------------------------------------- #
#  Reach funnel: build patients_reached & effect_size_mean from metrics       #
# --------------------------------------------------------------------------- #
FUNNEL_FACTORS = ("patient_fraction", "engaged_fraction", "adoption_fraction",
                  "counterfactual_share")
EFFECT_FACTORS = ("clinical_capture",)


def cost_effectiveness_funnel(
        cfg: Config | None = None, *, annual_budget: float = 100_000.0,
        annual_unique_visitors: float = 10_000.0,
        # each factor is a (median, sd) truncated normal, clipped to [0, 1]
        patient_fraction: tuple = (0.75, 0.10),
        engaged_fraction: tuple = (0.30, 0.10),
        adoption_fraction: tuple = (0.25, 0.10),
        counterfactual_share: tuple = (0.50, 0.12),
        clinical_capture: tuple = (0.50, 0.13),
        channel: str = "both",
        beneficiary_episodic_share: float | None = None,
        n_mc: int = 5_000,
        **overrides) -> dict:
    """Cost-effectiveness where `patients_reached` and `effect_size_mean` are BUILT
    UP from observable funnel factors (each a median+spread uncertainty) instead of
    typed in as point values. Each factor is a truncated normal (median, sd) clipped
    to [0, 1]; the ranges are propagated by Monte Carlo, returning p10/p50/p90 bands.

        patients_reached  = annual_unique_visitors
                            x patient_fraction        (P visitor is a CH patient)
                            x engaged_fraction        (P engaged | patient)
                            x adoption_fraction       (P tries a treatment | engaged)
                            x counterfactual_share    (would NOT have happened anyway)
        effect_size_mean  = clinical_capture          (realized share of full benefit)

    `patients_reached` is thus a COUNTERFACTUAL adopter count (people who tried a
    treatment because of us and would not have otherwise). `effect_size_mean` is
    then purely clinical_capture: of the full no-treatment -> full-access benefit,
    the share those adopters actually realize. Placing the counterfactual discount
    in reach (vs effect size) is an arbitrary but tidy choice; the product is what
    matters, and it must appear on exactly one side. The sim runs ONCE for the
    per-patient full benefit; outputs scale linearly in reach x effect, so the
    whole range is propagated cheaply.

    Funnel-factor sources: annual_unique_visitors and *_fraction come from PostHog
    (uniques, patient share via referrer mix, engagement via scroll/dwell/opens,
    adoption anchored by prints/return-visits or an opt-in follow-up cohort);
    counterfactual_share and clinical_capture come from the website survey (Q3, and
    Q7-Q9 severe-attack reduction)."""
    cfg = cfg or Config()
    if overrides:
        cfg = replace(cfg, **overrides)
    if channel not in CF_CHANNELS:
        raise ValueError(f"channel must be one of {CF_CHANNELS}")

    pp = _per_patient_benefit(cfg, channel, beneficiary_episodic_share)
    rng = np.random.default_rng(cfg.seed + 404)

    def draw(r):
        return _trunc_normal(rng, r[0], r[1], n_mc)

    reached = (annual_unique_visitors * draw(patient_fraction)
               * draw(engaged_fraction) * draw(adoption_fraction)
               * draw(counterfactual_share))
    eff = draw(clinical_capture)

    out = _metric_bands(pp, reached * eff, annual_budget)
    out.update({
        "annual_budget": annual_budget,
        "annual_unique_visitors": annual_unique_visitors,
        "channel": channel, "n_mc": n_mc,
        "beneficiary_episodic_share": beneficiary_episodic_share,
        "patients_reached": _bands(reached),
        "effect_size_mean": _bands(eff),
        "factors": {
            "patient_fraction": list(patient_fraction),
            "engaged_fraction": list(engaged_fraction),
            "adoption_fraction": list(adoption_fraction),
            "counterfactual_share": list(counterfactual_share),
            "clinical_capture": list(clinical_capture),
        },
    })
    return out


# --------------------------------------------------------------------------- #
#  Demo                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    res = simulate()
    for k, v in res.summary().items():
        print(f"  {k:38s}: {v:,.3f}" if isinstance(v, float) else f"  {k:38s}: {v:,}")

    print("\n--- Global attacks/yr by peak intensity ---")
    d = res.intensity_distribution("attacks")
    for i, c in enumerate(d, 1):
        print(f"  {i:2d}: {c:>15,.0f}  ({100*c/d.sum():4.1f}%)")

    print("\n--- Global person-years/yr spent at each intensity (within-attack profile) ---")
    tal = res.time_at_levels() / 60 / 24 / 365
    for i, c in enumerate(tal, 1):
        print(f"  {i:2d}: {c:>12,.0f} person-yr  ({100*c/tal.sum():4.1f}%)")
