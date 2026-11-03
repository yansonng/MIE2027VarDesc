# Supplementary methods and results

Supplement to *Can Patient-Specific Information Improve Medication-Based Risk Assessment? A Machine Learning Study of Bleeding-Related Inpatient Admissions* (MIE 2027). Variable definitions: [variables.md](variables.md).

## S1. Comparing Janusmed with the models at equal alert volumes ("flagging")

**Why this is needed.** Janusmed assigns each admission a bleeding-risk level (0, I, II or III); it gives no probability of an outcome. The machine learning models output a predicted probability for each admission. So the two cannot be compared on the same scale directly.

**Method.**
1. Each Janusmed level is turned into an alert rule. For example, "≥II" flags every admission at level II or III. In our cohort the rules flag:
   - ≥III: 6.6% of admissions;
   - ≥II: 16.1%;
   - ≥I: 23.7%.
2. Each model flags the same number of admissions: those with its highest predicted risk. This gives both approaches the same workload.
3. Each approach is scored by its **sensitivity**: the share of all bleeding-related admissions it flags.

Flagging admissions at random would catch, on average, the same share of bleeding-related admissions as the share of admissions flagged: 16.1% at the ≥II volume. That is the dashed "random flagging" line in Figure 1.

**Worked example at ≥II (16.1% of admissions, 7,650 of 47,608):**

| Approach | Admissions flagged | Bleeding-related admissions caught (sensitivity) |
|---|---|---|
| Random flagging (expected) | 16.1% | 16.1% |
| Janusmed rule ≥II | 16.1% | 18.3% (345 of 1,886) |
| Model 1 (medication), its top 16.1% | 16.1% | 20.6% |
| Model 2 (medication + patient), its top 16.1% | 16.1% | 41.4% |
| Model 3 (patient only), its top 16.1% | 16.1% | 41.3% |

At the same number of alerts, Model 2 catches 2.3 times as many bleeding-related admissions as the Janusmed rule.

**How this appears in the paper:**
- **Figure 1, right panel.** Each curve shows sensitivity (y-axis) for every possible alert volume (x-axis), with each model flagging its highest-risk admissions first. The dotted vertical lines mark the three Janusmed rules. The points are the comparisons at those volumes, with 95% CIs.
- **Table 1.** Each Janusmed level is compared on its own. Janusmed flags exactly the admissions at that level. Each model flags an equally large slice of its own ranking:
  - level III corresponds to the model's top 6.6%;
  - level II to the next 9.5%;
  - level I to the next 7.6%;
  - level 0 to the remaining 76.3%.

  The model columns give the share of all bleeding-related admissions that falls in each slice. Each column sums to 100%.

**Details.**
- Predicted probabilities that tie at the cut-off are split evenly, as under random tie-breaking.
- Out-of-fold predictions from the five cross-validation folds are pooled and ranked together.
- 95% CIs come from 2,000 patient-level (cluster) bootstrap resamples, with the fitted models held fixed. The rule's alert volume is recomputed in each resample, so the models always flag exactly as many admissions as the rule.
- Holm adjustment covers four tests: Model 1 − Janusmed and Model 2 − Model 1, each on AUROC and on sensitivity at ≥II. Adjusted p is <0.005 for all except Model 1 − Janusmed sensitivity at ≥II (0.031).

## S2. Choice of learner: XGBoost and logistic regression

XGBoost was chosen as a widely used learner for tabular clinical data. It handles missing values natively (for example, no haemoglobin measurement) and captures non-linear effects and interactions without manual specification.

To check that the conclusions do not depend on this choice, every input set was also fitted with a logistic regression. This L2-penalised regression (C = 1) uses cubic B-spline terms (five quantile knots) for age, haemoglobin, MCV and inpatient days, median imputation with missing-value indicators, and indicator variables for the Janusmed level. On the medication side, two sequence models over the medication history (a transformer and a GRU) were also fitted.

All models used the same five patient-grouped cross-validation folds and pooled out-of-fold predictions. The table gives AUROC (AUPRC; prevalence 0.040):

| Input variables | Logistic regression | XGBoost (paper) | Other |
|---|---|---|---|
| Janusmed level only | – | – | 0.512 (0.041), the level itself |
| Medication, 37 variables (Model 1) | 0.551 (0.048) | **0.552 (0.049)** | – |
| Extended medication, 51 variables¹ | 0.557 (0.048) | 0.552 (0.050) | tuned XGBoost 0.552 (0.049) |
| Medication history as a sequence | – | – | transformer 0.542 (0.050); GRU 0.550 (0.050) |
| Patient, 28 variables (Model 3) | 0.696 (0.127) | **0.699 (0.133)** | – |
| Medication + patient, 65 variables (Model 2) | 0.697 (0.127) | **0.700 (0.135)** | – |

¹ The 37 medication variables plus the raw (unbanded) Janusmed scores and five drug-pair terms.

**What the comparison shows:**
- **Both learners give the same picture.** Medication information alone stays near 0.55, and patient information raises the AUROC to about 0.70. Adding medication information to the patient variables adds little (see also S5).
- **XGBoost has similar AUROC and higher AUPRC when patient variables are included** (0.135 vs 0.127 for the combined inputs). On the extended medication set, logistic regression has the slightly higher AUROC (0.557 vs 0.552).
- **Logistic regression is somewhat better calibrated.** Calibration slope for the combined inputs is 0.92 for logistic regression vs 0.91 for XGBoost; for the patient inputs it is 0.95 vs 0.91.
- **More flexible representations of the medication history did not improve on Model 1:** neither the extended variables nor the sequence models.

## S3. Hyperparameters

All XGBoost models used the same fixed hyperparameters: 300 trees, maximum depth 4, learning rate 0.05, row and column subsampling 0.8, histogram tree method, and library defaults otherwise. There was no class weighting and no recalibration.

They were deliberately not tuned per model. The question is how much each set of input variables adds. Tuning each model separately would mix that question with how well each model happened to be optimised.

**Sensitivity analysis.** We checked whether tuning could have changed the medication side, where a better-optimised model would matter most for the comparison with Janusmed:
- **The model:** XGBoost on the extended medication variables, tuned with Optuna (TPE sampler, 30 trials) inside each training fold.
- **Validation:** the inner validation set was a separate, patient-disjoint training fold. The test fold was never used for tuning.
- **Objective:** AUPRC.
- **Search space:**
  - number of trees 100–800;
  - maximum depth 2–8;
  - learning rate 0.01–0.2;
  - row subsampling 0.5–1.0;
  - column subsampling 0.3–1.0;
  - minimum child weight 1–20;
  - L2 regularisation 0.001–10.

**Result.** Tuning changed neither AUROC (0.552 tuned vs 0.552 fixed; difference −0.000, bootstrap p = 0.99) nor AUPRC (0.049 vs 0.050; p = 0.62).

## S4. Temporal validation

**Design.** The models were trained once on admissions from 2020–2022 and tested on 2023, with the same fixed settings as in the main analysis and no tuning.

| Set | Admissions | Bleeding-related (%) | Patients |
|---|---|---|---|
| Training, 2020–2022 | 35,857 | 1,408 (3.9) | 24,197 |
| Test, all 2023 | 11,751 | 478 (4.1) | 9,237 |
| Test, new patients | 7,442 | 300 (4.0) | 6,271 |

Patients can appear in both periods, so "new patients" repeats the test on 2023 patients with no admission in 2020–2022 (63.3% of 2023 admissions).

The comparison at equal alert volumes works as in S1, using each test population's own Janusmed rule volumes:
- all 2023: 6.6% (≥III), 16.3% (≥II), 23.6% (≥I);
- new patients: 5.1%, 13.5%, 21.4%.

95% CIs and p-values come from 2,000 patient-level bootstrap resamples of the test population, with the fitted models held fixed, so they reflect test-set sampling only. This is a secondary analysis and p-values are not adjusted for multiple comparisons.

**Table S4.1. Discrimination and sensitivity at the volume of each Janusmed rule**

| Population | Approach | AUROC (95% CI) | Sensitivity at ≥III, % | Sensitivity at ≥II, % | Sensitivity at ≥I, % |
|---|---|---|---|---|---|
| All 2023 | Janusmed | 0.518 (0.497–0.539) | 6.7 (4.5–9.3) | 19.2 (15.7–23.1) | 27.2 (22.9–31.4) |
| All 2023 | Model 1 | 0.535 (0.508–0.562) | 10.3 (7.4–12.9) | 19.2 (15.8–22.9) | 28.6 (24.5–32.8) |
| All 2023 | Model 2 | 0.712 (0.686–0.737) | 27.6 (23.5–31.9) | 43.3 (38.5–47.5) | 51.5 (46.9–56.2) |
| All 2023 | Model 3 | 0.708 (0.680–0.733) | 28.9 (24.4–32.9) | 43.9 (38.7–48.3) | 52.1 (47.2–56.6) |
| New patients | Janusmed | 0.531 (0.505–0.559) | 4.7 (2.4–7.3) | 17.7 (13.5–22.1) | 27.7 (22.5–33.2) |
| New patients | Model 1 | 0.516 (0.484–0.549) | 7.7 (4.7–10.9) | 16.4 (12.3–20.6) | 26.3 (21.5–31.1) |
| New patients | Model 2 | 0.693 (0.662–0.725) | 22.7 (17.8–27.5) | 37.3 (31.4–42.7) | 47.0 (41.6–52.9) |
| New patients | Model 3 | 0.692 (0.658–0.725) | 23.7 (18.9–28.7) | 38.0 (31.0–43.6) | 47.7 (41.3–53.4) |

**Table S4.2. Comparison per Janusmed level, all 2023 (as Table 1 of the paper).** Last four columns: share of all bleeding-related admissions, % (95% CI), at that level or in an equally large band of each model's ranking.

| Janusmed level | Admissions, % | Bleeding-related, n (% of level) | Janusmed, % | Model 1, % | Model 2, % | Model 3, % |
|---|---|---|---|---|---|---|
| III | 6.6 | 32 (4.2) | 6.7 (4.5–9.3) | 10.3 (7.4–12.9) | 27.6 (23.5–31.9) | 28.9 (24.4–32.9) |
| II | 9.7 | 60 (5.2) | 12.6 (9.6–15.6) | 9.0 (6.8–12.0) | 15.7 (11.9–18.9) | 15.1 (11.6–18.6) |
| I | 7.3 | 38 (4.4) | 7.9 (5.5–10.7) | 9.3 (6.9–11.9) | 8.2 (6.1–11.3) | 8.2 (5.8–11.5) |
| 0 | 76.4 | 348 (3.9) | 72.8 (68.6–77.1) | 71.4 (67.2–75.5) | 48.5 (43.8–53.1) | 47.9 (43.4–52.8) |

**Table S4.3. Differences between approaches**

| Comparison | Population | AUROC difference (95% CI) | p | Sensitivity difference at ≥II, points (95% CI) | p |
|---|---|---|---|---|---|
| Model 1 − Janusmed | All 2023 | +0.017 (−0.014 to +0.048) | 0.30 | 0.0 (−4.3 to +4.3) | 1.00 |
| Model 1 − Janusmed | New patients | −0.016 (−0.055 to +0.022) | 0.41 | −1.3 (−6.4 to +3.8) | 0.67 |
| Model 2 − Model 1 | All 2023 | +0.176 (+0.145 to +0.205) | <0.002 | +24.1 (+18.7 to +28.6) | <0.002 |
| Model 2 − Model 1 | New patients | +0.177 (+0.140 to +0.216) | <0.002 | +20.9 (+14.6 to +26.8) | <0.002 |
| Model 2 − Model 3 | All 2023 | +0.004 (−0.008 to +0.016) | 0.51 | −0.6 (−3.8 to +2.4) | 0.74 |
| Model 2 − Model 3 | New patients | +0.001 (−0.015 to +0.018) | 0.89 | −0.7 (−4.1 to +3.4) | 0.96 |

**Calibration, all 2023.**

| Model | Calibration slope | O/E |
|---|---|---|
| Model 1 | 0.33 | 1.05 |
| Model 2 | 0.96 | 1.02 |
| Model 3 | 0.94 | 1.00 |

**Fixed thresholds.** We also fixed the probability cut-off that flags 16.0% of the training period (the 2020–2022 ≥II share, from out-of-fold training predictions) and applied it to 2023:

| Model | Flagged in 2023, % | Sensitivity, % |
|---|---|---|
| Model 1 | 14.8 | 18.4 |
| Model 2 | 17.1 | 43.7 |
| Model 3 | 17.3 | 45.2 |

**What this shows:**
- **The main finding holds.** Adding patient variables to Model 1 raised the AUROC by +0.176 in 2023 and +0.177 in new patients, and both CIs exclude zero. Model 2 again matches Model 3, so medication information adds little beyond the patient variables (see also S5).
- **Model 1's gain over Janusmed was not confirmed.** The 2023 difference is +0.017 (−0.014 to +0.048), and sensitivity at ≥II is the same (19.2%). In new patients the estimate is negative: −0.016 (−0.055 to +0.022). All these CIs include zero.
- **Calibration carried over for the models with patient variables,** but not for Model 1, whose slope of 0.33 means its predictions are too spread out.

**Limitations of this check.**
- It uses a single later year from the same region, so it is not external validation.
- The four undated utilisation variables (P25–P28 in [variables.md](variables.md)) may include information from the index visit, in the test year as in the main analysis.
- New patients have no admission in 2020–2022, so variable P24 (earlier bleeding-related admission) is almost never positive for them: under 0.1% of admissions, against 1.6% for all 2023.

## S5. Sensitivity analyses and outcome definition

**AUROC under alternative definitions** (cross-validated; 95% CIs from 2,000 patient-level bootstrap resamples).

| Analysis | Admissions | Bleeding-related | Janusmed | Model 1 | Model 2 | Model 3 | Model 2 − Model 1 (95% CI) |
|---|---|---|---|---|---|---|---|
| Main analysis | 47,608 | 1,886 | 0.512 | 0.552 | 0.700 | 0.699 | +0.148 (+0.131 to +0.164) |
| Without the four undated variables (P25–P28) | 47,608 | 1,886 | 0.512 | 0.552 | 0.699 | 0.683 | +0.147 (+0.130 to +0.162) |
| Patient records from the day before the visit excluded (windows start on day 2) | 47,608 | 1,886 | 0.512 | 0.552 | 0.688 | 0.684 | +0.136 (+0.118 to +0.152) |
| Both of the above | 47,608 | 1,886 | 0.512 | 0.552 | 0.682 | 0.668 | +0.130 (+0.113 to +0.146) |
| Bleeding-related admissions with main diagnosis D50, D61, D62, D64 or S06 removed (models refitted) | 46,812 | 1,090 | 0.527 | 0.543 | 0.658 | – | +0.115 (+0.095 to +0.136) |
| Gastrointestinal bleeding as the outcome, other bleeds removed | 46,275 | 553 | 0.543 | 0.557 | 0.687 | – | +0.130 (+0.097 to +0.161) |
| No recorded previous bleed | 42,957 | 1,375 | 0.506 | 0.547 | 0.656 | – | +0.109 (+0.090 to +0.127) |
| Admissions with listed medication only | 33,120 | 1,353 | 0.512 | 0.561 | 0.707 | 0.706 | – |

**What this shows:**
- **The gain from adding patient variables to Model 1 holds in every analysis,** although it is smaller when anaemia and traumatic diagnoses or patients with a previous bleed are removed.
- **Whether medication adds to patient variables depends on the undated variables.** In the main analysis Model 2 − Model 3 is +0.001 (−0.005 to +0.007). Without the undated variables it is +0.016 (+0.009 to +0.023), and in the 2023 test +0.008 (−0.008 to +0.023). Medication information therefore adds little, rather than nothing, once those variables are removed.
- **The undated variables are not a demonstrated leak.** The data export does not date them, so we cannot rule out that they include the index visit.

**Completeness of the medication list.** The list uses prescriptions (120 days) and recorded administrations (7 days). It excludes pharmacy pick-ups and multidose dispensing, because a pick-up usually repeats a prescription. However, a long-running prescription issued before the window can still be picked up inside it. Of the 14,488 admissions with no listed medication:
- 40.1% had a pharmacy pick-up in the 120 days before the visit;
- 25.6% had a multidose dispensing record in those 120 days;
- 65.2% had either;
- 74.5% had either, or a prescription 121–365 days before the visit.

Exposure and Janusmed levels are therefore probably underestimated, particularly at level 0.

**Outcome.** An admission is bleeding-related if its main diagnosis is in one of these ICD-10 code sets (a listed code matches itself and any longer code beginning with it):

| Group | ICD-10 codes |
|---|---|
| Gastrointestinal | I85.0, I98.3, K22.1, K22.6, K25.0–K25.2, K25.4–K25.6, K26.0–K26.2, K26.4–K26.6, K27.0–K27.6, K29.0, K62.5, K92.0–K92.2 |
| Intracranial | I60, I61, I62, S06.4, S06.5, S06.6 |
| Other sites | N93, R58, H11.3, H31.3, H35.6, H43.1, I31.2, I62.1, J94.2, K66.1, M25.0, R04.0, R04.8, R04.9, T81.0, N50.1A, R23.3W, T14.0A |
| Anaemia | D50, D61, D62, D64.9, D68.3, Y44.0 |

I62.1 is listed in both code sets; no admission in this cohort had that code.

**Positive predictive value at the ≥II alert volume** (share of flagged admissions that were bleeding-related): Janusmed 4.5%, Model 1 5.1%, Model 2 10.2%.
