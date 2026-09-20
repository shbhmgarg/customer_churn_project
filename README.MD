# Customer Churn Prediction

End-to-end machine learning project to predict customer churn for a
telecommunications company, plus a FastAPI REST API for serving live
predictions.

## Git Repository
https://github.com/shbhmgarg/customer_churn_project

## How to Run Locally (For Evaluators)

### Prerequisites
- Python 3.9 or higher
- pip (comes with Python)

### Step 1 - Extract the project folder
Unzip to any location.

### Step 2 - Open a terminal in the project folder
```
cd customer_churn_project
```

### Step 3 - Create a virtual environment (recommended)

Windows:
```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### Step 4 - Install dependencies
```
pip install -r requirements.txt
```

> **Note:** this project was trained and pickled using **scikit-learn
> 1.6.1**, pinned in `requirements.txt`. A version mismatch between
> training and deployment environments can break pickle loading — if you
> see an `InconsistentVersionWarning` or an `AttributeError` on startup,
> confirm your installed scikit-learn version matches the pin.

### Step 5 - Start the API
```
uvicorn app:app --reload
```

Expected output includes:
```
Uvicorn running on http://127.0.0.1:8000
```

Interactive docs are available at `http://127.0.0.1:8000/docs`.

### Step 6 - Test the API (in a new terminal)
```
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Step 7 - View the analysis notebook
```
pip install jupyter
jupyter notebook notebook/churn_analysis.ipynb
```
(Or open `notebook/churn_analysis.ipynb` directly in VS Code.)

### Troubleshooting
- **"No such file" for churn_model.pkl:** run commands from the project
  root, so the relative path `model/churn_model.pkl` resolves correctly.
- **Port 8000 already in use:** run
  `uvicorn app:app --reload --port 8001` instead.
- **scikit-learn version warning:** see the note in Step 4 above — pin
  the exact version listed in `requirements.txt`.

## Problem Statement

Customer churn directly impacts revenue. The goal is to identify customers
likely to churn so the retention team can proactively engage them before
they leave.

- **Dataset:** IBM Telco Customer Churn (7,043 customers, 21 columns)
- **Target:** `Churn` (binary — Yes / No)
- **Business priority:** Recall over Precision — missing a churner costs
  the full customer lifetime value; a false alarm only costs a single,
  low-cost retention outreach call.

## Project Structure

```
customer_churn_project/
├── data/
│   └── TelcoCustomerChurn.csv
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   └── churn_model.pkl
├── app.py
├── feature_engineering.py
├── requirements.txt
├── sample_request.json
└── README.md
```

## Workflow Summary

**1. Data Understanding & Preparation** — Loaded 7,043 rows × 21 columns.
Found 11 rows with a blank `TotalCharges` value, all with `tenure = 0`
(new customers not yet billed); converted to numeric and filled with 0
rather than dropped, since the value is logically known. No duplicate
rows or customer IDs. Split 70:30, stratified on `Churn`, `random_state=42`,
*before* encoding to avoid data leakage.

**2. EDA (8 visualizations)** — Key findings:
- Class imbalance: 26.5% churn overall
- Month-to-month contracts churn at ~43% vs. ~11% (one-year) and ~3%
  (two-year)
- Churn concentrated in the first ~12 months of tenure
- Fiber optic customers churn at ~42% vs. DSL's ~19%
- Higher `MonthlyCharges` correlates with higher churn
- No `TechSupport` correlates with higher churn
- `Electronic check` has the highest churn rate among payment methods
- `tenure` shows the strongest correlation with churn among numerical
  features

**3. Feature Engineering** — 4 new features:
- `TenureGroup` — tenure bucketed into New / Established / Loyal / Very
  Loyal
- `AvgMonthlySpend` — `TotalCharges / tenure` (falls back to
  `MonthlyCharges` when `tenure = 0`)
- `NumAdditionalServices` — count of add-on services subscribed to (0–6)
- `IsHighRiskProfile` — binary flag for customers who are both
  Month-to-month *and* pay by Electronic check

**4. Model Development** — Trained 4 Decision Tree configurations:

| Config | Settings | Train Acc | Test Acc | Churn Precision | Churn Recall | Churn F1 |
|--------|----------|-----------|----------|------------------|--------------|----------|
| 1 | Baseline (unrestricted) | 0.9980 | 0.7203 | 0.47 | 0.48 | 0.47 |
| 2 | max_depth=4 | 0.7980 | 0.7918 | 0.66 | 0.45 | 0.53 |
| **3 (final)** | max_depth=6, min_samples_leaf=20, class_weight='balanced' | 0.7629 | 0.7383 | 0.50 | **0.79** | 0.62 |
| 4 | Same as 3 + criterion='entropy' | 0.7629 | 0.7383 | 0.50 | 0.79 | 0.62 |

**Final model: Config 3.** Config 2 had higher raw accuracy, but only
caught 45% of actual churners. Config 3 trades some accuracy for nearly
double the recall (79%), which better serves the business goal of
proactively catching at-risk customers. Configs 3 and 4 were numerically
identical — switching the splitting criterion (Gini vs. entropy) made no
measurable difference on this dataset.

**5. Evaluation** — Confusion matrix (final model):

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 1117 (TN) | 435 (FP) |
| **Actual: Churn** | 118 (FN) | 443 (TP) |

Recall = 0.79 → the model catches roughly 4 out of every 5 actual
churners. Precision = 0.50 → about half of flagged customers are false
alarms, an acceptable tradeoff since a missed churner is more costly than
an unnecessary outreach call.

**6. Interpretation** — Top feature importances (from the Decision Tree):
- `Contract = Month-to-month` (~56.3%)
- `tenure` (~10.7%)
- `InternetService = Fiber optic` (~9.4%)
- `MonthlyCharges` (~4.8%)
- `TotalCharges` (~2.9%)

The tree's root split is on `Contract = Month-to-month`, confirming
contract type as the dominant churn signal — consistent with the EDA
findings. The 4 engineered features did not appear in the top 5; the raw
underlying fields proved most directly useful to this particular tree.

**7. Bonus Work**

Three bonus categories are suggested in the assignment brief: handling
class imbalance, hyperparameter tuning, and comparing additional models.
All three were done.

*Class imbalance handling* — `class_weight='balanced'` was applied in
Configs 3 & 4, and directly compared against the unweighted Configs 1 & 2
in the Model Development section above. The effect is large: Recall
roughly doubles (45% → 79%) once the minority (churn) class is weighted
properly, at the cost of some precision — the exact tradeoff the business
priority calls for.

*Hyperparameter tuning* — rather than relying only on the four
hand-picked configurations, a `GridSearchCV` search was run over
`max_depth` (4, 6, 8, 10, None), `min_samples_leaf` (5, 10, 20, 30, 50),
`criterion` (gini, entropy), and `class_weight` (balanced, None) — 100
combinations, each evaluated with 5-fold cross-validation, optimizing for
Recall. The search independently converged on `max_depth=6,
min_samples_leaf=20, criterion='gini', class_weight='balanced'` — exactly
Config 3 — with a cross-validated Recall of 0.7875, matching the held-out
test Recall of 0.79. This is a useful confirmation: the manual comparison
process in Model Development wasn't a lucky guess, it had already landed
on the true optimum for this hyperparameter space.

*Comparing additional models* — a Random Forest and a Logistic
Regression were trained on the same preprocessed data and compared
against the Decision Tree:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Decision Tree (Config 3, final) | 0.7383 | 0.50 | 0.79 | 0.62 |
| Random Forest (balanced) | 0.7466 | 0.52 | 0.80 | 0.63 |
| Logistic Regression (balanced) | 0.7379 | 0.51 | 0.79 | 0.62 |

Random Forest edges out the Decision Tree slightly on every metric, and
Logistic Regression performs almost identically despite not fully
converging (features weren't scaled, which its `lbfgs` optimizer is
sensitive to — scaling was intentionally skipped for a fast,
apples-to-apples comparison rather than building out a separate scaled
pipeline). **The Decision Tree (Config 3) remains the final model**, for
two reasons beyond the marginal performance gap: the assignment
specifically asks for a Decision Tree Classifier, and a single tree can be
visualized and explained node-by-node to a non-technical retention team,
which a 200-tree Random Forest ensemble cannot. Random Forest's small edge
is worth noting as a natural next step if the interpretability requirement
were relaxed in a future iteration.

**8. Model Serving** — FastAPI REST API at `POST /predict`.

## API Reference

### GET /
Returns a simple status message confirming the API is running.

### POST /predict
Request body: JSON with the 19 original customer fields (all columns
except `customerID` and `Churn`). Field types and allowed categorical
values are validated automatically via Pydantic.

**Response:**
```json
{
  "prediction": "Yes",
  "churn_probability": 0.6282
}
```

**Error handling:** invalid types, missing fields, or out-of-range
categorical values (e.g. `"InternetService": "Satellite"`) are rejected
automatically by FastAPI's request validation with a `422` response and
details on what failed, before any prediction logic runs. Unexpected
errors during prediction return a `400` with a descriptive message.

## Preprocessing Consistency

The API uses the exact same trained `Pipeline` object as the notebook —
not a reimplementation of the preprocessing steps. The pipeline bundles:

1. Feature engineering (`feature_engineering.py` — shared by both the
   notebook and the API, so there is exactly one implementation of this
   logic)
2. One-hot encoding (`ColumnTransformer` + `OneHotEncoder`, fitted only
   on the training set)
3. The trained Decision Tree classifier

Because all three steps are bundled into one saved object
(`churn_model.pkl`), calling `.predict()` on raw new customer data
guarantees identical preprocessing to what was used during training —
eliminating the risk of preprocessing drift between training and
serving.

## Reproducibility

All random seeds are set to 42. The train/test split is stratified to
preserve the ~26.5% churn rate in both folds.

## Key Business Takeaways

- **Contract type is the strongest churn predictor.** Converting
  month-to-month customers to annual contracts is likely the
  highest-leverage retention lever available.
- **New customers need attention.** Churn is heavily concentrated in the
  first 12 months of tenure.
- **Fiber optic customers churn more than DSL.** Worth investigating
  service quality or price sensitivity in that segment.
- **The model trades precision for recall.** We accept a moderate false
  positive rate (~50% of flagged customers are false alarms) because the
  cost of missing a real churner outweighs the cost of an unnecessary
  outreach call.

## Author

Shubham Garg