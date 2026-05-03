# heart-attack-detection-ml

Built this predictive model to analyze medical data and flag patients at high risk for heart attacks. Pushing the core ML pipeline here.

It's a machine learning pipeline that takes in patient vitals and EHR data (age, cholesterol, resting blood pressure, max heart rate, ECG results, etc.) and predicts the likelihood of heart disease.

I spent a lot of time on feature engineering — combining raw metrics into clinical risk indicators like the pressure_rate_ratio and an age_chol_risk index. This boosted the model accuracy significantly compared to just feeding raw features into the algorithm.

Trained two different models to compare approaches. Random Forest from scikit-learn is great for interpretability and I used it to extract feature importance, which showed that the engineered high_risk_profile and oldpeak (ST depression) were the strongest predictors. A 3-layer feed-forward neural network with dropout using TensorFlow/Keras slightly edged out the Random Forest in accuracy.

Both models hit around 83% accuracy on the test set.

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
python heart_disease_predictor.py
```

This generates a synthetic dataset (mimicking the UCI Heart Disease dataset structure), performs the feature engineering, trains both models, and outputs the accuracy metrics and charts to the outputs/ folder.
