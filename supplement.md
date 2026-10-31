# Supplementary methods and results

Supplement to *Can Patient-Specific Information Improve Medication-Based Risk Assessment? A Machine Learning Study of Bleeding-Related Hospital Admissions* (MIE 2027). Variable definitions: [variables.md](variables.md).

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
- 95% CIs come from 2,000 patient-level (cluster) bootstrap resamples. The rule's alert volume is recomputed in each resample, so the models always flag exactly as many admissions as the rule.

## S2. Choice of learner: XGBoost and logistic regression

XGBoost was chosen in advance as a widely used learner for tabular clinical data. It handles missing values natively (for example, no haemoglobin measurement) and captures non-linear effects and interactions without manual specification.

To check that the conclusions do not depend on this choice, every input set was also fitted with a logistic regression. This regression uses natural-spline terms for age, haemoglobin, MCV and inpatient days, and indicator variables for the Janusmed level. On the medication side, two sequence models over the medication history (a transformer and a GRU) were also fitted.

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
- **Both learners give the same picture.** Medication information alone stays near 0.55, and patient information raises the AUROC to about 0.70. Adding medication information to the patient variables adds nothing.
- **XGBoost has equal or slightly higher discrimination**, mainly in AUPRC (0.135 vs 0.127 for the combined inputs).
- **Logistic regression is somewhat better calibrated.** Calibration slope for the combined inputs is 0.92 for logistic regression vs 0.91 for XGBoost; for the patient inputs it is 0.95 vs 0.91.
- **More flexible representations of the medication history did not improve on Model 1:** neither the extended variables nor the sequence models.

## S3. Hyperparameters

All XGBoost models used the same hyperparameters, fixed before the analysis: 300 trees, maximum depth 4, learning rate 0.05, row and column subsampling 0.8, histogram tree method, and library defaults otherwise. There was no class weighting and no recalibration.

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
