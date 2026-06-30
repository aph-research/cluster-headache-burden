# Cluster Headache Monte Carlo Simulation — Distilled Parameters

> Extracted from *Notes on cluster headache quantification.txt*. This document keeps **only** what is needed to (a) generate a random patient and (b) simulate every attack they have in a year as a list of `(duration, max_pain_intensity)` tuples. Downstream "suffering metric" / log-scale-mapping material is summarized briefly at the end since it affects how the tuples are later *valued*, not how they are *generated*.

---

## 0. Simulation structure

- **Unit of output:** per attack → `(duration_minutes, max_intensity_0_to_10)`.
- **Per patient:** a full year of attacks (a list of such tuples).
- **Four patient archetypes** (2×2):
  - Episodic vs. Chronic
  - Treated vs. Untreated (treatment access is really a *spectrum* — model effectiveness as a range, not a binary, if desired)
- **Workflow:** define distributions per group → simulate each group → weight results by group prevalence → scale to global population via annual prevalence.

The two stratifying variables matter because:

| Variable | Affects frequency | Affects duration | Affects intensity |
|---|---|---|---|
| Episodic vs. Chronic | **Yes (by definition)** | Unclear / weak (chronic maybe slightly longer) | No clear evidence of a difference |
| Treated vs. Untreated | Possibly (via shorter/fewer bouts) | **Yes — treated shorter** | **Yes — treated lower** (but partly selection: mild attacks go untreated) |

---

## 1. Scope / prevalence (for global extrapolation)

- **1-year prevalence: 53 per 100,000 (0.053%)** — Fischera et al. (2007) meta-analysis; 95% CI 26–95. *(User's chosen anchor for global numbers.)*
- Lifetime prevalence: **124 per 100,000 (0.12%)** — Fischera (2007); Schindler & Burish (2022) lit review of 16 papers: 0.12% (95% CI 0.10–0.15%). "Adults of all ages and both sexes."
- Rule of thumb: **~1 in 1,000** people.
- Global headcount references used in the notes: OPIS uses 0.1% → **8.1 million** (lifetime-ish); the EA-forum draft cites **~3 million**; 53/100k × ~8.1 B ≈ **~4.3 million** in a given year. → *Decide which base applies to an "annual" simulation (see Questions).*
- **Sex ratio:** male:female ≈ **4.3:1** overall; **~15:1** in chronic cases (Fischera 2007). Episodic ~3:1–3.4:1 in clinic samples.
- Geography: prevalence roughly region-independent; weak trend toward higher prevalence in more northern latitudes; possibly lower in developing countries (sparse data). Not enough to model regionally.

---

## 2. Episodic vs. chronic split

- **~80% episodic / ~20% chronic** (Schindler & Burish 2022). Fischera eCH:cCH ratio = **6.0** (≈86%/14%). Pearson (2019): **78% episodic**.
- ~**15%** of patients transition between subtypes over time.
- **Caution:** tertiary headache-center samples oversample chronic (e.g., Gaul 2012: 31% chronic = selection bias). Population-representative split is ~80/20.
- The notes' working model used **episodic_fraction = 0.8**.

**Definitions (ICHD):**
- Episodic: attacks occur in **bouts** (weeks–months), separated by remissions > 3 months.
- Chronic: attacks year-round, no remission longer than 3 months.

---

## 3. Treated vs. untreated split

The notes derive a **global ~43% treated / ~57% untreated** estimate:

- EU baseline (Rossi et al. 2019): 46.8% complete access, 35.2% restricted, 18% lacking; only **47% of EU population has unrestricted access** to effective treatments.
- Extrapolated globally (developed ~15% / intermediate ~25% / developing ~60% of population):
  - Complete ≈ 19.3%, Restricted ≈ 33.25%, Lacking ≈ 47.45%
  - Assuming 50% of "restricted" + 15% of "lacking" still get access → **Treated ≈ 43%, Untreated ≈ 57%**.

**Treatment effectiveness is a spectrum** (Pearson et al. 2019, % reporting "complete or very effective"):
- Triptans **54%**, oxygen **54%** (high efficacy)
- Dihydroergotamine 25%, ergotamine/cafergot 17%, caffeine/energy drinks 17%, intranasal ketamine 14%
- Opioids 6%, intranasal capsaicin 5%, intranasal lidocaine 2% (low efficacy)

**Important nuance — access ≠ use per attack.** Even among patients *with* access, not every attack is treated (Snoer: abortive treatment used in **84.6%** of attacks; patients often skip mild attacks and treat only when an attack ramps up). Prophylactic ("preventive") treatment is separate and more common in chronic patients.

**Preventives are a separate FREQUENCY channel (now modeled).** Abortives truncate a single attack's *duration*; preventives cut *how many* attacks occur (shorter bouts / lower daily frequency). Modeled as: `preventive_access_fraction` on a preventive → `preventive_responder_mean` responder rate → responders' annual attack count cut by `preventive_responder_reduction_mean`. Default responder rate **0.42** = participant-weighted Rusanen 2022 across first/second-line preventives (verapamil 50%/n=1877, corticosteroids 55%/n=1177, lithium 31%, topiramate 25%, melatonin 25%). See `CH_simulation_sources.md` for the full table.

---

## 4. Frequency (attacks per year)

### Building blocks
- **Daily attack frequency during an active period** (ICHD: 1 every other day → 8/day):
  - Episodic: ~**1.1–3.1/day** (Cho median 1.1; Gaul mean 3.1, SD 2.1)
  - Chronic: ~**1.5–3.3/day** (Cho median 2.0; Gaul mean 3.3, SD 3.0)
  - Burish (retrospective, peak of cycle): **3.9/day** (SD 2.0) — overestimate, asks about cycle peak.
  - Maximal-pain patients report slightly more attacks/day (4.0 vs 3.5).
- **Bout (cluster period) duration (episodic):** Gaul **8.5 weeks** (SD 5.7); commonly 4–12 weeks; Cho ~4 weeks. Bahra mean 8.6 weeks.
- **Bouts per year (episodic):** Gaul **1.2/yr** (SD 1.1); most studies "~1/year"; Li: <1/yr 41.6%, 1/yr 37.0%, >1/yr 21.4%.
- **Remission periods (episodic):** highly variable; Rozen mean **~22 months** (SD ~27) — strongly right-skewed.
- **Chronic:** attacks essentially year-round (≈365 days), few/no remissions.

### Resulting annual totals
Cross-study estimates of **total attacks/year** (whole population): median ~**70**, mean pulled up to ~**300–400** by a heavy right tail (lognormal). Per-source summary:

| Source | Median /yr | Mean /yr |
|---|---|---|
| Gómez-Emilsson 2019 (survey, lognormal fit) | 50 | 614 (raw) / 165 (truncated) |
| Cho et al. 2019 | 78 | 175 |
| Rozen et al. 2001 | — | 120 |
| Pearson et al. 2019 | — | 262 |
| Li et al. 2022 | — | ~60–121 |

**The notes' final per-group model results** (best target to reproduce):

| Group | Attacks/yr mean | median | IQR |
|---|---|---|---|
| Episodic (treated & untreated similar) | ~200 | ~162 | 96–276 |
| Chronic (treated & untreated similar) | ~700 | ~655 | 474–902 |

→ Frequency is dominated by **episodic vs. chronic**, not by treatment. Distribution is **lognormal / long-tailed**. (Whether treatment reduces total attacks — via shorter or fewer bouts / lower daily frequency — is an open modeling choice; the notes tried adding a modest reduction for treated patients.)

**Modeling recommendation from notes:** simulate (distributions) for daily frequency and bout structure; can fix point estimates for episodic-fraction, bouts/year, bout duration.

---

## 5. Duration of attacks (minutes)

- **ICHD definition:** **15–180 minutes** (untreated).
- **Time from onset to peak:** ~**9 minutes** (maximal intensity within 9 min in **86%** of sufferers; Torelli mean 8.9 min, SD 9.5). Roughly constant regardless of total duration → onset/offset ramps are short.
- **Prospective duration data:**
  - Snoer 2018/2019 (prospective, mostly treated): overall **mean 66.3 min** (SD 94.2, range 5–750); **median treated 36.6, untreated 30.0** (untreated lower here because mild attacks go untreated — selection effect).
  - Cho 2019: episodic **median 60 min** (IQR 60–120); chronic **median 105 min** (IQR 60–172.5).
  - Rozen 2001: women **67.2 min**, men **88.2 min**.
  - Black 2005: prospective 230-pt study, min-duration avg **72 min** vs max-duration avg **159 min**.
- **Treated < untreated** (early abortive treatment shortens attacks), though prospective diaries blur this because patients self-select which attacks to treat.
- Duration is right-skewed → **lognormal** is a reasonable family.

### Intensity–duration coupling (more severe attacks last longer)
From Hagedorn (single chronic patient, 4,600 attacks), Claude-derived linear factor:
```
duration_factor = 0.1064 * intensity + 0.5797      (1.0 = average duration)
```
i.e. Mild ~29 min, Moderate ~39 min, Severe ~49 min, Very severe ~59 min. Supported by Russell (very-slight attacks significantly shorter).

---

## 6. Max intensity per attack (0–10 scale)

**Key methodological lesson:** *retrospective, all-things-considered* ratings (e.g., "how bad is a CH?") cluster at 9.7–10 because people recall their worst attack. *Prospective diaries* of individual attacks give **substantially lower and more spread-out** values. Use prospective data to generate per-attack max intensity.

| Source | Type | Max-intensity finding |
|---|---|---|
| Burish 2020 | Retrospective, ATC | **9.7 ± 0.6**; 72% rate 10/10 (use as ceiling reference, **not** per-attack) |
| Gaul 2012 | Retrospective | mean NRS **9** (SD 1) |
| Torelli 2003 | Prospective, **untreated**, episodic, first-half-of-bout | mean **9.17** (SD 1.0); 69% at 9–10, 86% at 8–10, 14% below 8 *(biased high: untreated + early bout)* |
| Snoer 2018/19 | **Prospective diary** | overall mean **7.0** (SD 2.3, range 1–10); **treated median 7.3** (IQR 5.9–8.7), **untreated median 7.0** (IQR 5.0–8.4) |
| Russell 1981 | Prospective, **untreated**, 5-pt scale, 77 attacks | very slight 12, slight 5, moderate 20, severe 17, extremely severe 23 |
| Hagedorn 2019 | Prospective, 1 chronic pt, **heavily treated** | mean **3.6** (SD 1.28); mild 14.2%, moderate 65.7%, severe 16.9%, very severe 3.2% |
| Snoer 2019 (patient-level) | Prospective | 38.5% of patients have mean severity ≥8; 61.4% have mean <8 |
| Cho 2019 | "Prospective" (no diary) | VAS median **9.0** (eCH IQR 8–10; cCH IQR 7.6–10) |
| Sohn | Prospective-ish | definite CH **9.1 ± 1.1**; probable CH 8.6 ± 1.8 |

**The notes' fitted per-group intensity model** (target to reproduce):

| Group | mean | median | SD / IQR |
|---|---|---|---|
| Untreated | **7.7** | **8.4** | IQR 6.1–9.7 |
| Treated | **6.4** | **6.4** | IQR 4.6–8.5 |

(Alternative earlier fits: untreated mean 6.8 / median 7.3 / SD 2.54; treated mean 6.9 / median 7.1 / SD 1.76.)

→ **Treatment lowers intensity; episodic vs. chronic shows no documented intensity difference.** Note the ~4% of "probable CH" patients rate <7/10; severe/very-severe is the vast majority.

**Note on bout position (episodic):** attacks are milder at the **start and end** of a bout, most severe in the **middle** — so per-attack max intensity can be modeled as rising then falling across a bout.

---

## 7. Within-attack intensity profile (for time-at-each-level, downstream)

Not needed to produce `(duration, max_intensity)` tuples, but needed if you later compute time spent at each pain level:
- Rapid rise to peak (~9 min), pain holds **at/near max** for most of the attack, then ends abruptly or fades quickly (may wax/wane or have super-intense stabs).
- Snoer phase structure: build-up ~10 min → attack/peak phase → resolution ~20 min.
- **Speculative** time-at-level (Claude estimate, no hard data): 10/10 ≈ 60–70%, 9/10 ≈ 20–25%, 8/10 ≈ 5–10%, ≤7/10 ≈ 0–5% of attack time. *Flagged as an assumption.*

---

## 8. Scale mapping (linear 0–10 → logarithmic suffering) — downstream only

Affects how tuples are *valued*, not generated. Core facts:
- Reported "most vs. 2nd-most painful experience ≈ 2×" (HTV) and CH 9.7 vs. labor 7.2 → the linear scale likely understates ratios; a 2-order-of-magnitude span is plausible (e.g., CH=100, labor=50).
- VAS shown to behave roughly **linearly/ratio** for mild-to-moderate and even severe acute pain (Myles 1999, 2005) — but contested.
- **Ceiling effect:** 72% rating 10/10 likely means the true distribution extends above the measurable ceiling; reconstruction techniques (Tobit/censored regression, mixture models, truncated-normal recovery) suggested.
- Andrés' suggestion: weight several candidate mappings (e.g., quadratic and exponential) by plausibility.

---

## 9. Open questions / modeling decisions to settle before coding

1. **Population base for "annual" run:** anchor on 1-yr prevalence 53/100k applied to current global population (~4.3 M), or on the 0.1% lifetime figure (~8.1 M)? Annual prevalence misses episodic patients with no bout this year — do we count them as zero-attack patients or exclude them?
2. **Max intensity: integer 1–10 or continuous?** (Output tuple format.)
3. **Treatment model:** binary 4-group, or continuous effectiveness range? Which channels does treatment act on — intensity only, duration only, frequency too? (Evidence: definitely intensity ↓ and duration ↓; frequency effect uncertain.)
4. **Access vs. per-attack use:** model treatment at the patient level (has access y/n) or per-attack (even patients with access leave ~15% of attacks untreated, skewing toward treating the worse ones)?
5. **Correlations to bake in:** intensity↔duration (positive, §5 formula), intensity↔frequency (weak positive, possibly via treatment), bout-position↔intensity (milder at bout edges)?
6. **Distribution families:** lognormal for frequency and duration (long-tailed) — confirm; intensity as truncated normal / mixture capped at 10?
7. **Within-attack profile & scale mapping:** treat as a *separate* later stage, or build in now? (I'd keep §7–§8 out of the tuple generator and apply them as a post-processing/valuation layer.)
