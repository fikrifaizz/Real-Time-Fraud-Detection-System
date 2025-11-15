"""
SHAP Explanation Helper
"""
import pandas as pd

def explain_prediction(transaction_data, model, explainer, top_n=5):
    """
    Explain a single transaction prediction
    """
    # Predict
    fraud_proba = model.predict_proba(transaction_data)[:, 1][0]

    # Get SHAP values
    shap_vals = explainer.shap_values(transaction_data)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Get top contributing features
    feature_contrib = pd.DataFrame({
        'feature': transaction_data.columns,
        'value': transaction_data.iloc[0].values,
        'shap_value': shap_vals[0]
    }).sort_values('shap_value', key=abs, ascending=False).head(top_n)

    # Format explanation
    risk_factors = []
    for _, row in feature_contrib.iterrows():
        risk_factors.append({
            'feature': row['feature'],
            'value': float(row['value']),
            'impact': 'increase' if row['shap_value'] > 0 else 'decrease',
            'shap_value': float(row['shap_value'])
        })

    return {
        'fraud_probability': float(fraud_proba),
        'prediction': 'FRAUD' if fraud_proba >= 0.5 else 'LEGITIMATE',
        'risk_level': 'HIGH' if fraud_proba >= 0.7 else 'MEDIUM' if fraud_proba >= 0.3 else 'LOW',
        'top_risk_factors': risk_factors
    }
