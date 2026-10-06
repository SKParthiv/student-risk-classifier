# Student Risk Classifier — TODO & Decision Log

DCS & GDG AI/ML recruitment task. Deadline: **Oct 7, 2026**.
Dataset: `dcs_student_data.csv` (10,030 rows, 21 cols).
Goal: classify students into **Low / Medium / High** academic risk and generate per-student recommendations.

## Known facts about this dataset (verified, don't re-derive)

- **No predictive signal exists.** All pairwise correlations between features and scores ≈ 0 (e.g. Attendance vs Final_Score = −0.014). `Total_Score` doesn't even correlate with its own component columns. `Grade` is random across the full score range. The data is synthetic with independent random columns.
  - Consequence: the EDA conclusion is a *proven negative*, stated plainly with a heatmap as evidence. The model section compares against a majority-class baseline honestly instead of pretending to find signal.
- **~2% dirty data is planted deliberately** (this is the real test):
  - ~200 NaNs each in Age, Attendance, Midterm, Final, Assignments, Quizzes, Participation, Projects
  - Gender: `' FEMALE '`, `' MALE '` (whitespace variants); Department: `' engineering '`, `'BUSINESS'`, `'Math'`, `'CS'` vs `'Computer Science'`
  - Invalid values: Age −3 and 87; Attendance −12 and 135; Final_Score −5 and 132; `math_score` contains a literal `'\t41'` string (column loads as text)
  - 30 duplicate Student_IDs

## Pipeline (ordered)

### Phase 1 — Parse-level cleaning (before ANY plot)
- [ ] Coerce all score columns with `pd.to_numeric(errors='coerce')` (fixes `'\t41'`)
- [ ] Strip whitespace + normalize case on Gender, Department; merge synonyms (`Math`→`Mathematics`, `CS`→`Computer Science`)
- [ ] Apply invalid-value conditions (see Decision Conditions below) → set out-of-range to NaN
- [ ] Drop duplicate Student_ID rows

### Phase 2 — EDA / Visualise (in `visualisation_scores.ipynb`)
- [ ] Correlation heatmap of all numeric columns (the money plot)
- [ ] Scatter: each score column vs Final_Score and Total_Score
- [ ] Boxplots: Total_Score by Department, by Gender
- [ ] Attendance vs marks scatter + correlation number (task explicitly asks for this)
- [ ] Markdown takeaway under EVERY plot — thought process is being graded
- [ ] Final markdown cell: state the no-signal finding plainly

### Phase 3 — Define risk criteria ← *the step the task grades hardest*
- [ ] Write the Low/Medium/High rule (see Decision Conditions) in a markdown cell with justification
- [ ] Apply rule → create `Risk` label column; print class distribution

### Phase 4 — Fill missing + split
- [ ] Median-impute numeric NaNs (median, not mean — planted garbage drags the mean)
- [ ] 80/20 stratified split on the `Risk` label (`stratify=y`)

### Phase 5 — Model
- [ ] Features: early-semester only (see Decision Conditions — no leakage)
- [ ] Baseline: `DummyClassifier(strategy='most_frequent')` — the number to beat
- [ ] Candidates: Logistic Regression, Decision Tree, Random Forest
- [ ] Train, then hyperparameters: start with RF defaults; tune `max_depth`, `n_estimators` only if train≫test gap appears

### Phase 6 — Evaluate
- [ ] Accuracy, per-class Precision/Recall, **macro-F1**, confusion matrix
- [ ] Overfit check: train F1 vs test F1 (train ≫ test → overfit → reduce depth)
- [ ] Report recall on HIGH class specifically (missing a struggling student is the costly error)
- [ ] Compare all models against the DummyClassifier baseline; state the honest conclusion

### Phase 7 — Output & bonus
- [ ] `predict_risk(student)` → prints the task's exact format (Student / Attendance / Marks / Risk / Recommendation)
- [ ] Recommendation rules: map (risk level, weakest factor) → specific advice string
- [ ] Bonus: auto-generate CSV report `Student | Attendance | Risk | Recommendation` for all rows

## Decision Conditions

**Invalid values (fix at parse time, before EDA):**
| Column | Condition | Action |
|---|---|---|
| any score, Attendance | value < 0 or > 100 | set NaN → later median-impute |
| Age | value < 15 or > 60 | set NaN → median-impute |
| math_score etc. | non-numeric string | `to_numeric(errors='coerce')` |
| Gender/Dept | leading/trailing space, case | strip + title-case + synonym map |
| Student_ID | duplicated | keep first, drop rest (before split — same student in train & test contaminates evaluation) |

**Missing values:** numeric → median. (Never mean with dirty data. ~2% missing, so imputation can't distort anything.)

**Feature set (no leakage):**
- IN: Attendance *(report-only, not a model feature — decided: zero correlation with marks)*, Midterm_Score, Assignments_Avg, Quizzes_Avg, Participation_Score, Projects_Score, math/reading/writing/science, test_preparation_course, Department (one-hot), Gender (one-hot)
- OUT: Final_Score, Total_Score, Grade — these *define* the label; feeding them in is re-deriving the answer

**Risk label criteria (proposed default — adjust thresholds, then commit):**
- High: Total_Score < 60
- Medium: 60 ≤ Total_Score < 75
- Low: Total_Score ≥ 75
- (Attendance < 60% noted in the recommendation text regardless of label, since it's excluded from features)

**Model choice:** Random Forest unless logistic regression matches it — trees need no scaling and handle outliers natively. But expect all models ≈ baseline; that result, shown cleanly, IS the deliverable.

**Metric priority:** macro-F1 > HIGH-class recall > accuracy. Justify in one markdown line (imbalanced classes; false negatives cost most).

**Overfit condition:** if train F1 − test F1 > 0.1 → reduce `max_depth`, retrain, recheck.

## Definition of done
- [ ] Notebook runs top-to-bottom clean, comments explain *why*, not what
- [ ] Every plot has a written takeaway
- [ ] The no-signal finding is stated explicitly with heatmap evidence
- [ ] Metrics include the baseline comparison
- [ ] Prediction output matches the task's example format
- [ ] Bonus report CSV generated
