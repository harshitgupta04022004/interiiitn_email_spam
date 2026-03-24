import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessTrainingData import processData, processDataForEvaluation
from classifier import get_xgb_model, evaluate_model, random_search_xgb
import json
import pickle
import os

def train():
    """
    Train the email spam classifier model.
    
    Steps:
    1. Process training data
    2. Split into train/test sets
    3. Perform hyperparameter tuning with RandomizedSearchCV
    4. Save model and feature columns
    5. Evaluate model performance
    
    Returns:
        Trained XGBoost model
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "TRAINING EMAIL SPAM CLASSIFIER")
    print("=" * 70)
    
    # Process data
    print("\n[1/5] Processing training data...")
    X, y = processData()
    
    # Split data
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.1, random_state=42, stratify=y
    )
    print(f"   Train size: {X_train.shape}")
    print(f"   Test size: {X_test.shape}")
    print(f"   Class distribution in train: {np.bincount(y_train)}")
    
    # Train model with hyperparameter tuning
    print("\n[3/5] Training model with hyperparameter tuning...")
    print("   This may take several minutes...")
    xgb_model = random_search_xgb(X_train, y_train)
    
    # Save artifacts
    print("\n[4/5] Saving model artifacts...")
    os.makedirs('artifacts', exist_ok=True)
    
    # Save model
    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("   ✓ Model saved to artifacts/model.pkl")
    
    # Save feature columns
    with open('artifacts/feature_columns.json', 'w') as f:
        json.dump(list(X.columns), f)
    print("   ✓ Feature columns saved to artifacts/feature_columns.json")
    print(f"   ✓ Total features: {len(X.columns)}")
    
    # TLD encoders are already saved in preprocessTrainingData.py
    print("   ✓ TLD encoders saved to artifacts/tld_encoders.pkl")
    
    # Evaluate model
    print("\n[5/5] Evaluating model on test set...")
    y_pred = xgb_model.predict(X_test)
    
    print("\n" + "=" * 70)
    print(" " * 25 + "MODEL PERFORMANCE")
    print("=" * 70)
    evaluate_model(y_pred, y_test)
    
    print("\n" + "=" * 70)
    print(" " * 20 + "TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nYou can now use the model for predictions:")
    print("  python gradioInterface.py")
    print("=" * 70 + "\n")
    
    return xgb_model


def predict(sender_name, sender_email, receiver_email, subject, body):
    """
    Predict spam type for a single email.
    
    Args:
        sender_name: Name of the sender
        sender_email: Email address of sender
        receiver_email: Email address of receiver  
        subject: Email subject line
        body: Email body text
        
    Returns:
        String with classification result formatted in Markdown
    """
    # Check if model files exist
    if not os.path.exists('artifacts/model.pkl'):
        return """
**❌ Error: Model Not Found**

The trained model file is missing. Please train the model first by running:

```bash
python train.py
```

This will train the model and save it to `artifacts/model.pkl`.
"""
    
    if not os.path.exists('artifacts/feature_columns.json'):
        return """
**❌ Error: Feature Columns Not Found**

The feature columns file is missing. Please train the model first by running:

```bash
python train.py
```
"""
    
    if not os.path.exists('artifacts/tld_encoders.pkl'):
        return """
**❌ Error: TLD Encoders Not Found**

The TLD encoder file is missing. Please train the model first by running:

```bash
python train.py
```
"""
    
    try:
        # Process input data
        X = processDataForEvaluation(sender_name, sender_email, receiver_email, subject, body)

        # Load feature columns
        with open('artifacts/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)

        # Align features (handle missing columns from training)
        X = X.reindex(columns=feature_cols, fill_value=0)

        # Load model
        with open('artifacts/model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Make prediction
        y_pred = loaded_model.predict(X.values)
        y_proba = loaded_model.predict_proba(X.values)

        # Get prediction results
        prediction = int(y_pred[0])
        confidence = float(y_proba[0][prediction])

        # FIXED: Better label formatting with emojis
        labels = {
            0: "✅ Ham (Legitimate Email)", 
            1: "⚠️ Spam", 
            2: "🎣 Phishing Attack", 
            3: "💰 Nigerian 419 Scam"
        }
        
        # Get label details
        label_descriptions = {
            0: "This email appears to be legitimate correspondence.",
            1: "This email contains spam content (advertisements, promotions, etc.).",
            2: "**WARNING:** This email appears to be a phishing attempt designed to steal credentials or personal information.",
            3: "**WARNING:** This email appears to be a Nigerian 419 scam (advance-fee fraud)."
        }

        # Format output with confidence score
        result = f"""
### **Classification Result**

**Category:** {labels[prediction]}

**Confidence:** {confidence * 100:.2f}%

**Description:** {label_descriptions[prediction]}

---

**All Probabilities:**
- Ham: {y_proba[0][0] * 100:.2f}%
- Spam: {y_proba[0][1] * 100:.2f}%
- Phishing: {y_proba[0][2] * 100:.2f}%
- Nigerian 419: {y_proba[0][3] * 100:.2f}%
"""
        
        print(f"\n✓ Prediction: {prediction} - {labels[prediction]} (Confidence: {confidence*100:.2f}%)")
        return result
        
    except FileNotFoundError as e:
        return f"""
**❌ Error: File Not Found**

{str(e)}

Please ensure you have trained the model by running:
```bash
python train.py
```
"""
    except Exception as e:
        return f"""
**❌ Error During Prediction**

An unexpected error occurred:

```
{str(e)}
```

Please check:
1. All required files exist in the `artifacts/` directory
2. The input data is valid
3. You have trained the model with the same feature set

If the problem persists, retrain the model:
```bash
python train.py
```
"""


if __name__ == "__main__":
    # Train the model when script is run directly
    model = train()
