# Heart disease detection

I rebuilt this project from scratch. The old version generated
synthetic patients from hand written distributions, made the label
out of a risk formula written in the same script, and then reported
how well the models learned that formula as if it were clinical
performance. It also called the target heart attack prediction,
which it never was. This version uses the real UCI Heart Disease
data, names the target correctly, and runs the comparisons the old
version only talked about.

## What the target actually is

The dataset is the processed Cleveland file from the UCI Heart
Disease dataset (dataset 45): 303 patients seen at the Cleveland
Clinic, 13 clinical features (age, sex, chest pain type, resting
blood pressure, cholesterol, and others), and a target "num" from
0 to 4 for angiographic disease status. I binarise it: num above 0
means disease present. 297 rows remain after dropping rows with
missing values, and 46.1% of them are positive.

So this predicts the dataset's disease status label. It does not
predict heart attacks, it is not trained on my patients, and it is
not a medical tool.

## Models and results

Stratified 80/20 split, seed 42, fixed before fitting. Every model
also gets 5 fold stratified cross validation, because with 297
rows a single 60 patient test set is easy to fool yourself with.

| Model | Test accuracy | Test ROC AUC | CV accuracy |
|---|---|---|---|
| Random Forest, raw features | 0.867 | 0.943 | 0.815 |
| Random Forest, engineered features | 0.867 | 0.934 | 0.811 |
| Logistic regression, engineered | 0.833 | 0.949 | 0.825 |
| MLP neural network, engineered | 0.833 | 0.930 | 0.771 |

Full numbers in `outputs/metrics.json`.

## The feature engineering question, answered honestly

I added three engineered features: age times cholesterol, resting
blood pressure over max heart rate, and a high risk flag for
atypical chest pain with exercise induced angina. The old version
claimed these boosted accuracy significantly. It never ran the
comparison. I ran it: same split, same model, raw against
engineered. Test accuracy is identical (0.867) and cross validation
is a touch worse with the engineered features (0.811 against
0.815). On this dataset, my engineered features do not help. The
code keeps them, the README keeps the result, and the claim is
gone.

Top features in the engineered Random Forest: thal (0.127), ca
(0.108), cp (0.102).

## The neural network

The MLP here is a scikit-learn feed forward network, two hidden
layers (64 and 32). The old version used Keras and claimed it
edged out the Random Forest. Here the MLP is the weakest model by
cross validation (0.771). On 297 rows that is not a surprise, and
I am reporting it as measured.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/run_analysis.py
python -m pytest tests/
```

## Disclaimer

Educational and research use only. Not a medical device, not
validated for clinical use, and it must not be used to diagnose
anyone. See a qualified medical professional for anything real.

## Files

- `src/run_analysis.py` loads, splits, trains all four model
  variants, and writes `outputs/metrics.json`.
- `EXPLAIN.md` explains the setup and the small data problem.
