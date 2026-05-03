"""
Heart Attack Detection System
Machine learning model for analyzing medical data.
Feature engineering, scikit-learn, TensorFlow/Keras.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def generate_medical_data(n_samples=5000):
    """Generate synthetic medical dataset based on UCI Heart Disease features."""
    print(f"Generating synthetic medical dataset ({n_samples} records)...")
    np.random.seed(42)
    
    # Features commonly found in heart disease datasets
    age = np.random.normal(55, 10, n_samples).clip(25, 90).astype(int)
    sex = np.random.choice([0, 1], n_samples, p=[0.3, 0.7]) # 1 = male
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.16, 0.28, 0.09]) # chest pain type
    trestbps = np.random.normal(130, 15, n_samples).clip(90, 200) # resting blood pressure
    chol = np.random.normal(240, 40, n_samples).clip(120, 400) # cholesterol
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15]) # fasting blood sugar > 120
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.50, 0.02]) # resting ECG
    thalach = np.random.normal(150, 20, n_samples).clip(70, 200) # max heart rate
    exang = np.random.choice([0, 1], n_samples, p=[0.67, 0.33]) # exercise induced angina
    oldpeak = np.random.exponential(1.0, n_samples).clip(0, 6) # ST depression
    
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak
    })
    
    # Feature Engineering (simulating the resume claim)
    print("Performing feature engineering...")
    df['age_chol_risk'] = (df['age'] / 50) * (df['chol'] / 200)
    df['pressure_rate_ratio'] = df['trestbps'] / df['thalach']
    df['high_risk_profile'] = ((df['age'] > 60) & (df['chol'] > 250) & (df['trestbps'] > 140)).astype(int)
    
    # Target variable generation (simulating realistic correlations)
    risk_score = (
        (df['age'] / 100) * 1.0 +
        df['sex'] * 0.2 +
        (df['cp'] == 0).astype(int) * 1.5 + # Typical angina is actually less risky in some datasets
        (df['trestbps'] / 200) * 0.8 +
        (df['chol'] / 400) * 0.5 +
        df['exang'] * 1.2 +
        (df['oldpeak'] / 6) * 1.5 +
        df['high_risk_profile'] * 1.0
    )
    
    # Add noise
    risk_score += np.random.normal(0, 0.5, n_samples)
    
    # Target: 1 = Heart Disease, 0 = No Heart Disease
    threshold = np.median(risk_score)
    df['target'] = (risk_score > threshold).astype(int)
    
    return df

def train_models(df):
    """Train and evaluate ML models."""
    os.makedirs('outputs', exist_ok=True)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Random Forest (Scikit-learn)
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    # Feature Importance
    feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp.values, y=feature_imp.index)
    plt.title('Feature Importance for Heart Attack Prediction')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    # 2. Deep Learning Model (TensorFlow/Keras)
    print("\nTraining Deep Neural Network (Keras)...")
    tf.random.set_seed(42)
    
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    loss, nn_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Neural Network Accuracy: {nn_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/training_history.png')
    plt.close()
    
    # Save metrics
    with open('outputs/model_metrics.json', 'w') as f:
        json.dump({
            "dataset_size": len(df),
            "features_used": len(X.columns),
            "engineered_features": ["age_chol_risk", "pressure_rate_ratio", "high_risk_profile"],
            "random_forest_accuracy": round(rf_acc, 4),
            "neural_network_accuracy": round(nn_acc, 4),
            "top_3_features": feature_imp.index[:3].tolist()
        }, f, indent=4)
        
    print("\nTraining complete. Check 'outputs/' directory.")

if __name__ == "__main__":
    df = generate_medical_data()
    train_models(df)
