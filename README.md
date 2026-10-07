# Student Risk Classifier — DCS & GDG AI/ML Task

Identifying students at academic risk from `dcs_student_data.csv` (10,030 rows, 21 columns).

**Headline finding:** the dataset is synthetically generated with independent random
columns — it contains *no learnable signal* (proved below, linearly and empirically).
The deliverable is therefore an explicit, auditable **criteria-based risk system**
(what the task asks for: *"define your own criteria ... and explain your approach"*),
plus a complete ML due-diligence notebook proving why a learned model cannot beat a
baseline on this data.

## Repository map

| File | Contents |
|---|---|
| `01_data_cleaning.ipynb` | Parse-level cleaning of the planted dirty data → `dcs_student_data_cleaned.csv` |
| `02_eda.ipynb` | Exploration: heatmap, scatters, boxplots, attendance analysis → the no-signal conclusion |
| `03_risk_system.ipynb` | **The answer:** risk criteria, per-student prediction output, recommendation engine, bonus report |
| `04_ml_models_due_diligence.ipynb` | Baseline + LogReg / Decision Tree / Random Forest — the honest negative result |
| `student_risk_recommendations.csv` | Bonus report: `Student \| Attendance \| Risk \| Recommendation` for all 10,000 students |

## 1. Data Exploration

- **Every pairwise correlation between numeric columns is ≈ 0** (all |r| < 0.02).
  Attendance vs Final_Score: **r = −0.014**. Attendance vs Total_Score: flat as well,
  and mean score is constant across all 5% attendance bands.
- `Total_Score` is uniform on [50, 100] and independent even of its own component
  columns (`Midterm`, `Final`, `Assignments`, `Quizzes`, `Projects`) — impossible in
  real data, diagnostic of synthetic generation.
- `Grade` is meaningless: every grade A–F spans the full 50–100 Total_Score range.
- Boxplots of Total_Score by Department and by Gender overlap almost perfectly.

**Conclusion (stated in `02_eda.ipynb` with plots as evidence):** no feature, numeric or
categorical, carries information about any score. EDA on this dataset is a proven negative.

## 2. Data Preprocessing

The raw file contains deliberately planted dirt (~2% of cells), handled in `01_data_cleaning.ipynb`:

| Dirt | Examples | Handling |
|---|---|---|
| Missing values | ~200 NaNs in each of 8 columns | Kept as NaN for EDA; **median-imputed inside the ML pipeline** (fit on train only — no leakage) |
| Non-numeric strings | `'\t41'` in `math_score` | `pd.to_numeric(errors='coerce')` |
| Inconsistent labels | `' FEMALE '`, `' engineering '`, `'Math'`, `'CS'` | strip, normalise case, merge synonyms |
| Impossible values | Age −3 and 87; Attendance −12 and 135; Final_Score −5 and 132 | Set to NaN (outside [0,100] for scores, [15,60] for age) |
| Duplicate IDs | 30 duplicated Student_IDs | Dropped (before splitting — same student in train+test contaminates evaluation) |

Result: 10,030 → **10,000 clean rows**.

## 3. Risk Model — explicit criteria

| Risk | Criterion | Share of cohort |
|---|---|---|
| **High** | Total_Score < 60 | 1,955 students (~20%) |
| **Medium** | 60 ≤ Total_Score < 75 | 3,035 (~30%) |
| **Low** | Total_Score ≥ 75 | 5,010 (~50%) |

**Why criteria-based and not learned:** the label must come from defensible rules (the
task's own requirement), and `04` proves no classifier can learn those rules from the
features — because the features are independent of everything, including the score the
rules are built on. Attendance is deliberately not in the label (zero information in this
data) but low attendance (< 60%) is still surfaced in every recommendation.

## 4. Evaluation (from `04_ml_models_due_diligence.ipynb`)

| Model | Accuracy | Macro-F1 | High-risk recall | Train macro-F1 | Overfit gap |
|---|---|---|---|---|---|
| Dummy (majority class) | 0.501 | 0.223 | 0.00 | 0.223 | 0.00 |
| Logistic Regression | 0.501 | 0.223 | 0.00 | 0.223 | 0.00 |
| Decision Tree | 0.371 | 0.325 | 0.20 | 1.000 | 0.67 |
| Random Forest | 0.496 | 0.254 | 0.01 | 1.000 | 0.75 |
| Random Forest (max_depth=8) | 0.501 | 0.223 | 0.00 | 0.240 | 0.02 |

**Interpretation:** models either collapse exactly onto the baseline (learn "always Low")
or memorise the training set (train F1 = 1.0) and return to baseline on test. The
Decision Tree's higher macro-F1 comes with *below-baseline accuracy* and is within noise
on ~2,000 test rows — a louder coin flip, not a better model. This is the expected
behaviour when label ⊥ features, and no feature engineering can change it
(any deterministic transformation of independent variables stays independent).

## 5. Prediction & Recommendation

`predict_risk(student_id)` in `03_risk_system.ipynb` prints the task's required format:

```
Student: Omar Williams (S1000)
Attendance: 52%
Marks: 56
Risk: HIGH
Recommendation: Immediate intervention: targeted remediation in science score
with weekly tutoring and mentor check-ins. Priority: raise attendance above 60%
(currently 52%).
```

Recommendations are driven by the student's **weakest factor**, compared *after
normalising each column by its scale* — `Participation_Score` is 0–10, all other scores
0–100, so raw comparison would flag Participation for nearly everyone. The resulting
advice varies across 10 distinct factors.

## 6. Bonus — automated report

`student_risk_recommendations.csv`: `Student | Attendance | Risk | Recommendation` for
all 10,000 students, generated from the explicit criteria (ground truth), not from a
model guessing at noise.

## What data would make this predictable

Real early-warning systems are built on features this dataset lacks: attendance *trend*
across the semester (not one aggregate), assignment submission lateness, LMS/login
activity, prior-year or entry scores, and midterm-to-final *delta*. With those, the
pipeline in `04` — leakage-free split, in-pipeline imputation, baseline-first evaluation —
is exactly the one you'd run, and it would find the signal immediately.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib
# Run the notebooks in order: 01 -> 02 -> 03 (04 is independent, needs 01's output)
```
