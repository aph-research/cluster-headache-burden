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
    time-to-relief), NOT lower the peak (peak ~9 min precedes the abortive;
    Snoer treated 7.3 vs untreated 7.0). `treated_peak_intensity_reduction`=0.
  * Per-attack peak comes from PROSPECTIVE diaries (mean ~7), NOT the
    ceiling-loaded retrospective 9.7/72%-at-10 (Burish) -- that is a recalled
    "worst-ever" rating, not a per-attack measurement.
  * PREVENTIVES act on a separate channel from abortives: they cut attack
    FREQUENCY (annual attack count), via shorter bouts (episodic) or lower daily
    frequency (chronic), NOT per-attack duration or peak. Modelled as a
    responder rate + a fractional frequency cut among responders.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields
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
    n_patients: int = 20_000                      # simulated cohort size
    seed: int = 0

    # ---- Subtype split ---------------------------------------------------- #
    episodic_fraction: float = 0.80               # Schindler&Burish; Fischera 6:1

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
    placebo_abort_prob: float = 0.18              # self/placebo abort (Cohen, suma RCTs)

    # ---- Treatment: effect magnitudes ------------------------------------- #
    aborted_duration_mean_min: float = 15.0       # time-to-relief (suma ~7-10, O2 by 15)
    aborted_duration_sd_min: float = 6.0
    aborted_duration_floor_min: float = 3.0
    treated_peak_intensity_reduction: float = 0.0 # fraction in [0,1]; default 0 (see docstring)

    # ---- Preventive treatment: reduces attack FREQUENCY ------------------- #
    # A preventive (verapamil first-line; corticosteroids transitional; lithium,
    # topiramate, melatonin, galcanezumab) lowers HOW MANY attacks a patient has
    # -- via shorter bouts (episodic) or lower daily frequency (chronic). Modelled
    # as a fractional cut to the patient's ANNUAL attack count, independent of
    # abortive access. Acts on frequency only (NOT per-attack duration or peak).
    preventive_access_fraction: float = 0.23      # bottom-up HIC/MIC/LIC (see sources)
    preventive_responder_mean: float = 0.42       # P(responds | on preventive); Rusanen 2022
    preventive_responder_reduction_mean: float = 0.55  # freq cut among responders
    preventive_responder_reduction_sd: float = 0.20
    preventive_reduction_floor: float = 0.50      # responder == >=50% reduction (trial convention)
    preventive_reduction_cap: float = 0.95

    # ---- Per-attack PEAK INTENSITY: two-level model (NRS 1..10) ----------- #
    intensity_mean: float = 7.0                   # population mean peak (Snoer ~7.0)
    intensity_between_sd: float = 1.6             # between-patient severity SD
    intensity_within_sd: float = 1.0             # within-patient attack-to-attack SD
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
    treated = has_access[patient_idx] & (rng.random(total) < cfg.treat_fraction)
    roll = rng.random(total)
    aborted = np.where(treated, roll < efficacy[patient_idx], roll < cfg.placebo_abort_prob)

    abort_dur = np.clip(
        rng.normal(cfg.aborted_duration_mean_min, cfg.aborted_duration_sd_min, total),
        cfg.aborted_duration_floor_min, None)
    duration = np.where(aborted, np.minimum(duration, abort_dur), duration)

    if cfg.treated_peak_intensity_reduction > 0:
        red = np.round(intensity * (1 - cfg.treated_peak_intensity_reduction))
        intensity = np.where(aborted, np.clip(red, 1, 10), intensity).astype(int)

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
#  Counterfactual access export (for cost-effectiveness analysis)             #
# --------------------------------------------------------------------------- #
def counterfactual_csv(cfg: Config | None = None, **overrides) -> str:
    """Per-(currently-untreated)-patient baseline attacks AND the model's
    counterfactual 'with access' attacks, under three interventions:
    abortive-only, preventive-only, and both.

    Use case: estimate what a nonprofit averts by giving an untreated patient
    access to treatment. Unlike a flat '% fewer attacks', this captures the real
    shape of the benefit -- abortives truncate DURATION (same attack count),
    preventives remove whole ATTACKS, neither changes the PEAK.

    Each scenario emits the attack-tuple list in the SAME format as grouped_csv
    (`(dur_int,intensity.2f)` joined by ';'), so a spreadsheet's own DLES /
    severity / time formulas can be applied to each scenario column directly;
    averted = (baseline metric) - (scenario metric). Convenience averted columns
    (attacks / minutes / severe>=9) are included too.

    All patients are generated as currently UNTREATED (the beneficiary pool); the
    counterfactual is that same patient gaining access. Additionality / how many
    beneficiaries you reach belong in the spreadsheet, not here."""
    import io
    cfg = cfg or Config()
    if overrides:
        cfg = replace(cfg, **overrides)

    # baseline cohort: nobody has access
    base = simulate(replace(cfg, treatment_access_fraction=0.0,
                            preventive_access_fraction=0.0))
    n = cfg.n_patients
    rp = np.random.default_rng(cfg.seed + 101)   # preventive draws
    ra = np.random.default_rng(cfg.seed + 202)   # abortive draws

    # per-patient: preventive responder + frequency cut; abortive efficacy
    responder = rp.random(n) < cfg.preventive_responder_mean
    reduction = np.where(
        responder,
        np.clip(rp.normal(cfg.preventive_responder_reduction_mean,
                          cfg.preventive_responder_reduction_sd, n),
                cfg.preventive_reduction_floor, cfg.preventive_reduction_cap),
        0.0)
    eff = np.clip(ra.normal(cfg.abort_prob_mean, cfg.abort_prob_sd, n), 0.0, 0.95)

    offs = np.concatenate(([0], np.cumsum(base.n_attacks)))
    dur, inten = base.duration, base.intensity

    def tup(d, it):
        return ";".join(f"({int(round(float(x)))},{float(y):.2f})"
                        for x, y in zip(d, it))

    def dles(d, it):
        # Days Lived in Extreme Suffering = minutes in >=9/10 pain / 60 / 24.
        # Whole duration of any attack peaking >=9 counts (matches the spreadsheet
        # convention; the dashboard's time_at_levels() uses the finer profile).
        return float(d[it >= 9].sum()) / 1440.0

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "id", "type", "total_attacks", "total_duration",
        "severe_attacks_baseline", "dles_baseline", "attacks",
        "attacks_with_access_abortive", "attacks_averted_abortive",
        "time_averted_abortive_min", "severe_averted_abortive", "dles_averted_abortive",
        "attacks_with_access_preventive", "attacks_averted_preventive",
        "time_averted_preventive_min", "severe_averted_preventive", "dles_averted_preventive",
        "attacks_with_access_both", "attacks_averted_both",
        "time_averted_both_min", "severe_averted_both", "dles_averted_both",
    ])
    for i in range(n):
        a, b = int(offs[i]), int(offs[i + 1])
        d = dur[a:b].astype(float)
        it = inten[a:b]
        nb = len(d)
        base_min = float(d.sum())
        sev_base = int((it >= 9).sum())
        dles_base = dles(d, it)

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

        def scen(d_use, it_use):
            return (tup(d_use, it_use), nb - len(d_use),
                    int(round(base_min - float(d_use.sum()))),
                    sev_base - int((it_use >= 9).sum()),
                    round(dles_base - dles(d_use, it_use), 5))

        ab = scen(d_trunc, it)            # abortive only
        pv = scen(d[keep], it[keep])      # preventive only
        bo = scen(d_trunc[keep], it[keep])  # both
        w.writerow([
            i, "episodic" if base.is_episodic[i] else "chronic", nb,
            int(round(base_min)), sev_base, round(dles_base, 5), tup(d, it),
            *ab, *pv, *bo,
        ])
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  Demo                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    res = simulate()
    for k, v in res.summary().items():
        print(f"  {k:38s}: {v:,.3f}" if isinstance(v, float) else f"  {k:38s}: {v:,}")

    print("\n--- Global attacks/yr by PEAK intensity ---")
    d = res.intensity_distribution("attacks")
    for i, c in enumerate(d, 1):
        print(f"  {i:2d}: {c:>15,.0f}  ({100*c/d.sum():4.1f}%)")

    print("\n--- Global person-years/yr spent AT each intensity (within-attack profile) ---")
    tal = res.time_at_levels() / 60 / 24 / 365
    for i, c in enumerate(tal, 1):
        print(f"  {i:2d}: {c:>12,.0f} person-yr  ({100*c/tal.sum():4.1f}%)")
