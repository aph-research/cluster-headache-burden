# Provenance for `ch_simulation.py` defaults

Every `Config` lever, its default value, and the evidence behind it. Tagged
**[measured]** (study data), **[definitional]** (ICHD-3 diagnostic boundary, *not*
a central tendency), **[pooled]** (meta-analysis/review), or **[assumption]**
(our modelling choice / weakly-evidenced). Confidence noted where relevant.

---

## Scope / scaling
| Lever | Default | Evidence |
|---|---|---|
| `annual_prevalence_per_100k` | 53 | **[pooled]** Fischera 2007 meta-analysis, 1-yr prevalence 53/100k (95% CI 26–95), for **adults of all ages, both sexes**. Lifetime 124/100k (0.12%). |
| `adult_population` | 5.8e9 | **[derived]** 53/100k is an ADULT (18+) prevalence, so it must be applied to the adult population (~72% of 8.1B), NOT the full world population. → **~3.07M** sufferers/yr (this falls out of prevalence × adult population; not tuned to any prior figure). |
| `episodic_fraction` | 0.80 | **[measured/cited]** ~80–90% episodic (Wei 2018 review; Vikelis 2016 cohort 77.5%). Fischera eCH:cCH ≈ 6:1. Clinic samples oversample chronic (selection bias). |

## Frequency
ICHD-3 criterion D: **1 every other day to 8/day [definitional]** — a ceiling, *not* typical.
Measured central tendency is far lower.
| Lever | Default | Evidence |
|---|---|---|
| `e_attacks_per_day_mean` | 2.0 | **[measured]** episodic median 1/day (Vikelis 2016, n=302); means 1.1–3.1 across studies (Cho, Gaul). |
| `c_attacks_per_day_mean` | 2.5 | **[measured]** chronic median 2/day (Vikelis 2016); Gaul mean 3.3. |
| `e_bout_weeks_mean` (SD) | 8.5 (5.7) | **[measured]** Gaul 2012. Review avg 6–12 wk; Vikelis median 4–6 wk. |
| `e_bouts_per_year_mean` (SD) | 1.2 (1.1) | **[measured]** Gaul 2012; most studies "~1/yr". |
| `c_active_fraction` | 0.90 | **[assumption]** chronic = remissions <3 mo by definition; ~year-round. |
| Resulting attacks/yr | episodic ~132, chronic ~820 | built bottom-up from the per-day/bout/bouts-per-year clinical params above (Gaul, Vikelis); not tuned to any target. Bounded by ICHD limits (≤8/day; episodic active ≤39 wk). |

## Duration (minutes, untreated/intrinsic)
ICHD-3: **15–180 min [definitional]**.
| Lever | Default | Evidence |
|---|---|---|
| `duration_median_min` | 45 | **[measured]** prospective mean 39.3 min (Hagedorn 2019, 4,600 attacks, n=1 patient). 825-pt cohort (Göbel 2021): mode 30, 54% in 15–60, ~90% within 15–180, mean 123 (recall-inflated, skewed). |
| `duration_sigma` | 0.70 | **[assumption]** lognormal shape giving right tail to ~180+. |
| `dur_intensity_slope` | 0.08 | **[assumption, lit-supported]** factor = 1 + slope·(intensity − mean), CENTERED on the population mean so median duration stays calibrated; clipped [0.3, 2.5]. Russell 1981 found a positive intensity↔duration correlation; Hagedorn implies slope ~0.1 but is n=1 over a narrow range, so default is gentler (0.08). **slope=0 ⇒ no coupling — vary in sensitivity.** (Earlier 0.106/0.58 form was Hagedorn-centered at intensity 3.6, which inflated durations ~32% at our mean of 7 — fixed.) |
| Retrospective overstates duration | — | **[measured]** Snoer 2019: untreated attacks rated significantly longer retrospectively than in diaries. |

## Peak intensity (NRS 1–10, intrinsic/untreated)
**The 9.7/10, 72%-at-10 figure (Burish 2021) is a retrospective, recalled "how bad is
your CH" rating — ceiling-loaded. It is NOT per-attack and must NOT be the sampling
distribution.** Use prospective diaries instead.
| Lever | Default | Evidence |
|---|---|---|
| `intensity_mean` / `intensity_between_sd` / `intensity_within_sd` | 7.0 / 1.6 / 1.0 | **[measured mean + assumed split]** TWO-LEVEL model: each patient draws a latent severity ~N(7.0, 1.6); their attacks vary ~N(patient, 1.0); rounded/clipped to 1–10. Marginal mean ~7, SD ~1.9, ~10% reach true 10. Mean anchor: Snoer 500 attacks ~7.0 (SD ~2.3). The between/within **split is an assumption** — aggregated studies can't separate them; Snoer found within-patient variability is LOW, so between dominates. Other anchors: Russell 1981 untreated (broad spread, ~22% mild); Hagedorn mean 3.6 (single heavily-treated patient — *not* generalizable); Torelli 9.17 (selection-biased high). |
| Time-to-peak | ~9 min | **[measured]** Torelli 2003 (8.9 min, 86% within 9 min). Used implicitly (peak precedes abortive). |

## Treatment
**Key finding [measured]: an effective abortive ABORTS the attack (truncates DURATION
to the time-to-relief); it does NOT lower the peak intensity.** Snoer 2019 prospective:
treated peak 7.3 ≈ untreated 7.0. Peak is reached (~9 min) before the abortive acts.
Hence `treated_peak_intensity_reduction` defaults to 0.

| Lever | Default | Evidence |
|---|---|---|
| `treatment_access_fraction` | 0.18 | **[measured HIC + inferred LMIC]** Global fraction with real access to an effective abortive. Independent estimate **~0.18** (sensitivity range 0.10–0.30): HIC ~0.55 (Rossi 2020: EU 47% unrestricted, 63–66% reimbursed; Evers & Rapoport 2017: O₂ reimbursed in 50% of mostly-HIC countries), MIC ~0.12, LIC ~0.03, pop-weighted (~85% LMIC). Bounded above by WHO neurological treatment-gap data (>50% MIC, >75% LIC) + low LMIC triptan consumption + pooled 10.4-yr diagnostic delay (J Headache Pain 2025). Only the HIC input is measured; LMIC values are inferences. **First lever to vary in sensitivity.** |
| `abort_prob_mean` (SD) | 0.60 (0.22) | **[pooled]** patient-level responder rates (Rusanen 2022): oxygen 65%, triptans 64%, SC sumatriptan ~78%, psilocybin 67%. |
| `treat_fraction` | 0.85 | **[measured]** ~85% of attacks treated even with access; mild ones skipped (Snoer). |
| `placebo_abort_prob` | 0.18 | **[measured]** acute-RCT placebo pain-free at 15 min: SC suma 17% (Cochrane), oxygen 20% (Cohen 2009). |
| `aborted_duration_mean_min` (SD) | 15 (6) | **[measured]** time-to-pain-free: SC sumatriptan ~7 min lag then seconds (Hardebo 1993); oxygen 78% pain-free by 15 min (Cohen 2009 JAMA, vs 20% air). |
| `treated_peak_intensity_reduction` | 0.0 | **[measured-justified]** see above; lever if you want to model a peak effect. |

### Abortive RCT effect sizes (reference; underlie `abort_prob`/`aborted_duration`)
| Treatment | Pain-free | Relief (no/mild) | Time | Source |
|---|---|---|---|---|
| SC sumatriptan 6 mg | 48% @15min | 75% @15min | ~7 min | Sumatriptan CH Study Group NEJM 1991; Cochrane CD008042 |
| High-flow O₂ 12 L/min | 78% @15min | — | by 15 min | Cohen 2009 JAMA (vs 20% air) |
| Intranasal sumatriptan 20 mg | 47% @30min | 57% @30min | — | van Vliet 2003 |
| Intranasal zolmitriptan 10 mg | 48% @30min | 62% @30min | — | Cittadini 2006; Rapoport 2007; Cochrane |
| Oral zolmitriptan 10 mg | — | 47% @30min (episodic only) | — | Bahra 2000 (ineffective in chronic) |
| Octreotide SC | — | 52% vs 36% placebo @30min | slow | Matharu 2004 |
| Intranasal lidocaine | — | ~27% moderate relief | slow (~37 min) | Robbins 1995; Costa 2000 |
| Opioids | no efficacy data; not recommended (AHS) | — | — | — |

### Preventives (not yet in model; would reduce attack FREQUENCY/bout length)
Responder rates [pooled, Rusanen 2022]: LSD 79%, psilocybin 69%, ergolines 61%,
corticosteroids 55%, verapamil 50%, lithium 31%, melatonin/topiramate 25%, propranolol 9%.
Verapamil is first-line preventive (Leone 2000 RCT). Galcanezumab reduces episodic
weekly frequency (Goadsby 2019 NEJM). Psilocybin RCT (Schindler 2022, n=14) was
statistically **negative** (−3.2 attacks/wk, p=0.25); the "~50% reduction" is an
uncontrolled within-subject extension (n=10). Survey "85% abort" (Sewell 2006) is
retrospective. → Psychedelic efficacy is NOT RCT-established; treat as low-confidence.

## Within-attack intensity profile — `time_at_levels()`
Converts "minutes by *peak* intensity" into "minutes actually *spent* at each level"
(needed for an honest time-at-10/10). Two models:

- **DEFAULT — fixed-minutes** (`profile_rise_min` 9, `profile_decline_min` 20):
  rise-to-peak ≈ 9 min [measured: Torelli time-to-peak 8.9 min], decline ≈ 20 min
  [measured: Snoer 2018 resolution phase ~20 min]; plateau = duration − rise − decline
  (≥0), so short/aborted attacks have little/no plateau. Linear ramps spread time
  uniformly across levels 1..peak. **Derived from primary data, independent of any
  prior burden model.**
- **COMPARISON ONLY — fractional 15/70/15** (`profile_*_frac`): the 70%-plateau split
  from a prior published burden model. Shown side-by-side in the UI for contrast; **not**
  used for headline numbers. It attributes ~70% of every attack to the peak, which
  overstates time-at-10/10 by ~1.2× vs the fixed-minutes model.

No cohort has measured a true intra-attack intensity time-series, so both are models;
the fixed-minutes one is built bottom-up from measured phase durations.
