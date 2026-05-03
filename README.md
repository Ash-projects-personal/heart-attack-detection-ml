# heart-attack-detection-ml

Built this predictive model to analyze medical data and flag patients at high risk for heart attacks. Pushing the core ML pipeline here.

## What this does

It's a machine learning pipeline that takes in patient vitals and EHR data (age, cholesterol, resting blood pressure, max heart rate, ECG results, etc.) and predicts the likelihood of heart disease/heart attack.

I spent a lot of time on feature engineering — combining raw metrics into clinical risk indicators like the `pressure_rate_ratio` and an `age_chol_risk` index. This boosted the model accuracy significantly compared to just feeding raw features into the algorithm.

I trained two different models to compare approaches:
1. **Random Forest (scikit-learn)**: Great for interpretability. I used this to extract feature importance, which showed that the engineered `high_risk_profile` and `oldpeak` (ST depression) were the strongest predictors.
2. **Deep Neural Network (TensorFlow/Keras)**: A 3-layer feed-forward network with dropout for regularization. Slightly edged out the Random Forest in accuracy.

## The numbers

- **Random Forest Accuracy**: ~83%
- **Neural Network Accuracy**: ~83%
- **Dataset**: 5,000 patient records with 13 features

## How to run

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
python heart_disease_predictor.py
```

This generates a synthetic dataset (mimicking the UCI Heart Disease dataset structure), performs the feature engineering, trains both models, and outputs the accuracy metrics and charts to the `outputs/` folder.

## Files

- `heart_disease_predictor.py`: The main ML pipeline
- `outputs/feature_importance.png`: Random Forest feature importance chart
- `outputs/training_history.png`: Keras DNN loss/accuracy curves
- `outputs/model_metrics.json`: Final evaluation metrics
