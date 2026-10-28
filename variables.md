# Variables used in the models

Supplementary material for *Can Patient-Specific Information Improve Medication-Based Risk Assessment? A Machine Learning Study of Bleeding-Related Hospital Admissions* (MIE 2027).

The models used 65 variables: 37 medication-related and 28 patient-related.

| Model | Variables |
|---|---|
| Janusmed | Janusmed bleeding-risk level only (variable M37) |
| Model 1: medication | the 37 medication-related variables (M1–M37) |
| Model 2: medication + patient | all 65 variables (M1–M37 and P1–P28) |
| Model 3: patient | the 28 patient-related variables (P1–P28) |

**Unit of analysis.** One row per hospital admission from the emergency care unit (ECU), Region Kalmar County, Sweden, 2020–2023.

**Timing.** Every variable uses information recorded before the day of the ECU visit (the index date), except the four healthcare-utilisation variables P25–P28. The data export does not date those four.

**Missing values.** Flags and counts are 0 when nothing was recorded. Laboratory values (P10, P11) and the prescription share (M21) are left missing when there is no measurement or no medication. XGBoost handles missing values natively.

## Medication-related variables (37)

**The medication list** for an admission is the set of distinct ATC codes (at least 5 characters) from:
- prescriptions in the 120 days before the index date;
- medications administered in hospital in the 7 days before the index date.

About 30% of admissions had no medication record. Their medication variables are 0, and their Janusmed level is 0.

### Descriptors of the medication list (21)

| # | Variable | Description | Values |
|---|---|---|---|
| M1 | `polypharmacy` | Number of distinct ATC codes on the medication list | count |
| M2 | `flag_anticoagulant` | Any antithrombotic agent, ATC B01A (includes antiplatelets) | 0/1 |
| M3 | `flag_antiplatelet` | Any platelet aggregation inhibitor, ATC B01AC | 0/1 |
| M4 | `flag_nsaid` | Any non-steroidal anti-inflammatory drug, ATC M01A | 0/1 |
| M5 | `flag_corticosteroid` | Any systemic corticosteroid, ATC H02 | 0/1 |
| M6 | `flag_ssri` | Any selective serotonin reuptake inhibitor, ATC N06AB | 0/1 |
| M7 | `atc_A` | Number of distinct codes in ATC main group A, alimentary tract and metabolism | count |
| M8 | `atc_B` | … group B, blood and blood-forming organs | count |
| M9 | `atc_C` | … group C, cardiovascular system | count |
| M10 | `atc_D` | … group D, dermatologicals | count |
| M11 | `atc_G` | … group G, genito-urinary system and sex hormones | count |
| M12 | `atc_H` | … group H, systemic hormonal preparations (excluding sex hormones and insulins) | count |
| M13 | `atc_J` | … group J, anti-infectives for systemic use | count |
| M14 | `atc_L` | … group L, antineoplastic and immunomodulating agents | count |
| M15 | `atc_M` | … group M, musculo-skeletal system | count |
| M16 | `atc_N` | … group N, nervous system | count |
| M17 | `atc_P` | … group P, antiparasitic products, insecticides and repellents | count |
| M18 | `atc_R` | … group R, respiratory system | count |
| M19 | `atc_S` | … group S, sensory organs | count |
| M20 | `atc_V` | … group V, various | count |
| M21 | `pharmacy_ratio` | Share of the medication list that comes from prescriptions rather than from hospital administrations | 0–1 (missing if no medication) |

### Janusmed outputs (16)

All 16 were computed from the medication list with the Janusmed knowledge base: Janusmed riskprofil for the risk categories, Janusmed interaktioner for the interactions.

**How the risk-category levels (M22–M30) are computed:**
- each substance's risk value for the category is summed over the list (for seizure risk, the maximum is taken);
- the result is converted to Janusmed's level 0–III (0 none, I somewhat, II moderate, III significant risk).

M22–M30 use each substance's peroral (systemic) risk value. M37 uses the value for each medication's actual drug form.

| # | Variable | Description | Values |
|---|---|---|---|
| M22 | `anticholinergic` | Anticholinergic effects, risk level | 0–3 |
| M23 | `bleeding_risk` | Bleeding risk level, peroral values for all substances | 0–3 |
| M24 | `constipation` | Constipation, risk level | 0–3 |
| M25 | `orthostatism` | Orthostatism, risk level | 0–3 |
| M26 | `qt_prolongation` | QT prolongation, risk level | 0–3 |
| M27 | `renal_toxicity` | Renal toxicity, risk level | 0–3 |
| M28 | `sedation` | Sedation, risk level | 0–3 |
| M29 | `seizure_risk` | Seizure threshold lowering, risk level | 0–3 |
| M30 | `serotonergic` | Serotonergic effects, risk level | 0–3 |
| M31 | `interaction_count_a` | Drug–drug interactions of class A (probably no clinical significance) | count |
| M32 | `interaction_count_b` | Drug–drug interactions of class B (clinical significance unknown or variable) | count |
| M33 | `interaction_count_c` | Drug–drug interactions of class C (clinically significant, can be handled, e.g. by dose adjustment) | count |
| M34 | `interaction_count_d` | Drug–drug interactions of class D (clinically significant, should be avoided) | count |
| M35 | `total_interactions` | Total number of drug–drug interactions (M31–M34) | count |
| M36 | `total_risk_burden` | Sum of the nine risk-category levels (M22–M30) | 0–27 |
| M37 | `level_form` | **Janusmed bleeding-risk level**, form-aware: each medication's risk value is taken for its actual drug form (peroral; rectal/sublingual/buccal; parenteral/transdermal/implant; local). This is the Janusmed classification the models are compared with. | 0–3 |

## Patient-related variables (28)

| # | Variable | Description | Window before the index date | Values |
|---|---|---|---|---|
| P1 | `age` | Age at the ECU visit | – | years |
| P2 | `sex_male` | Male sex | – | 0/1 |
| P3 | `prior_renal_disease` | Recorded renal disease | 2 years | 0/1 |
| P4 | `prior_cancer` | Recorded cancer | 2 years | 0/1 |
| P5 | `prior_liver_disease` | Recorded liver disease | 2 years | 0/1 |
| P6 | `prior_transplantation` | Recorded organ transplantation | 2 years | 0/1 |
| P7 | `prior_alcohol` | Recorded alcohol-related condition | 2 years | 0/1 |
| P8 | `prior_surg_procedure` | Recorded surgical procedure | 2 years | 0/1 |
| P9 | `prior_other_anemi` | Recorded other anaemia | 2 years | 0/1 |
| P10 | `hemoglobin_min` | Lowest blood haemoglobin | 2 years | g/L (missing if not measured) |
| P11 | `mcv_min` | Lowest mean corpuscular volume | 2 years | fL (missing if not measured) |
| P12 | `pacd_gi_bleeding` | Diagnosis in the gastrointestinal bleeding code set | 1–365 days | 0/1 |
| P13 | `pacd_cns_bleeding` | Diagnosis in the intracranial bleeding code set | 1–365 days | 0/1 |
| P14 | `pacd_other_bleeding` | Diagnosis in the other bleeding code set | 1–365 days | 0/1 |
| P15 | `pacd_general_bleeding` | Diagnosis in the eye bleeding code set | 1–365 days | 0/1 |
| P16 | `pacd_anemia_markers` | Diagnosis in the anaemia code set | 1–365 days | 0/1 |
| P17 | `pacd_cancer` | Cancer diagnosis | 1–365 days | 0/1 |
| P18 | `pacd_kidney` | Kidney disease diagnosis | 1–365 days | 0/1 |
| P19 | `pacd_transplantation` | Transplantation | 1–365 days | 0/1 |
| P20 | `pacd_liver` | Liver disease diagnosis | 1–365 days | 0/1 |
| P21 | `pacd_alcohol` | Alcohol-related diagnosis | 1–365 days | 0/1 |
| P22 | `pacd_liver_kidney` | Liver or kidney disease diagnosis | 1–365 days | 0/1 |
| P23 | `pacd_surgery_7d` | Surgery | 1–7 days | 0/1 |
| P24 | `prior_bleeding_historical` | Earlier bleeding-related hospital admission in the study cohort | 181–730 days | 0/1 |
| P25 | `rub` | Resource utilisation band (ACG system), healthcare-utilisation level | undated | 0–5 |
| P26 | `ic_care_days_1y` | Inpatient care days in the past year | undated | days |
| P27 | `ic_care_episodes_120d` | Inpatient care days in the past 120 days | undated | days |
| P28 | `ic_care_episodes_30d` | Inpatient care days in the past 30 days | undated | days |

**Notes**
- **P3–P11** come from the regional EHR event records. **P12–P23** come from the predefined diagnosis and procedure code sets of the regional data warehouse.
- **Overlap between the two groups.** Some conditions appear in both groups with different sources and look-back windows: P3–P9 look back 2 years, P12–P23 look back 1–365 days. The visit day itself is excluded throughout.
- **P27 and P28** keep their source names, which say "episodes", but the values are day counts.
- **P25–P28** are not dated in the data export, so they may include information from the index visit itself. A sensitivity analysis without these four variables gave the same conclusions: the AUROC gain from adding patient variables was 0.147 (95% CI 0.130–0.162).
