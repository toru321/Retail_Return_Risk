import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def run_retail_return_pipeline(data_path: str):
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # 2. Separate Features and Target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 3. Stratified Train-Test Split (Prevent Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Feature Scaling (Fit on Train, Transform on Test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Baseline Logistic Regression
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train_scaled, y_train)
    
    # 6. Model Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 7. Compute Test Performance Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    # 8. Feature Importance / Odds Ratios
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0],
        'Odds_Ratio': np.exp(model.coef_[0])
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    
    # 9. 5-Fold Cross-Validation Analysis
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = cross_validate(
        model, scaler.fit_transform(X), y, cv=cv, 
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    
    return metrics, cm, coef_df, cv_metrics

if __name__ == "__main__":
    metrics, cm, coef_df, cv_metrics = run_retail_return_pipeline('dataset_10_retail_return_risk.csv')
    
    print("=== MODEL TEST METRICS ===")
    for k, v in metrics.items():
        print(f"{k:<12}: {v:.4f}")
        
    print("\n=== CONFUSION MATRIX ===")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    
    print("\n=== FEATURE EFFECT ANALYSIS ===")
    print(coef_df.to_string(index=False))