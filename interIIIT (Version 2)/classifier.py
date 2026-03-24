from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def get_xgb_model():
    """
    Get base XGBoost model with optimized parameters.
    
    Returns:
        XGBClassifier with base configuration
    """
    return XGBClassifier(
        tree_method='hist',
        device='cuda',                  # Use GPU if available
        objective='multi:softprob',     # Multi-class classification
        num_class=4,                    # 4 classes: ham, spam, phishing, nigerian
        eval_metric='mlogloss',         # Multi-class log loss
        n_jobs=3,
        random_state=42,
        verbosity=1,
        # Additional parameters for better performance
        enable_categorical=False,        # We handle encoding manually
    )


def random_search_xgb(X_train, y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        
    Returns:
        Best XGBoost model after hyperparameter tuning
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH")
    print("=" * 60)
    
    # Get base model
    model = get_xgb_model()

    # IMPROVED: Better hyperparameter ranges based on best practices
    param_dist = {
        "n_estimators": randint(100, 300),        # More estimators for better learning
        "max_depth": randint(4, 10),              # Deeper trees to capture complexity
        "learning_rate": uniform(0.01, 0.15),     # Lower learning rate, more iterations
        "subsample": uniform(0.7, 0.3),           # Sample 70-100% of data
        "colsample_bytree": uniform(0.7, 0.3),    # Sample 70-100% of features
        "gamma": uniform(0, 3),                    # Minimum loss reduction
        "min_child_weight": randint(1, 7),         # Minimum sum of instance weight
        "reg_alpha": uniform(0, 0.5),              # L1 regularization
        "reg_lambda": uniform(1, 3),               # L2 regularization
        "scale_pos_weight": uniform(1, 2)          # Handle class imbalance
    }

    print("\nHyperparameter Search Space:")
    for param, dist in param_dist.items():
        if hasattr(dist, 'a') and hasattr(dist, 'b'):  # uniform distribution
            print(f"  {param}: [{dist.a:.2f}, {dist.a + dist.b:.2f}]")
        elif hasattr(dist, 'low') and hasattr(dist, 'high'):  # randint distribution
            print(f"  {param}: [{dist.low}, {dist.high}]")

    # Configure RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,                      # Try 50 different combinations
        scoring='f1_weighted',          # Optimize for weighted F1 score
        cv=3,                           # 3-fold cross-validation
        verbose=2,                      # Show progress
        n_jobs=3,                       # Use 3 parallel jobs
        random_state=42,
        refit=True                      # Refit on full training set with best params
    )

    print("\nStarting hyperparameter search...")
    print("This will take several minutes...")
    
    # Fit the search
    search.fit(X_train, y_train)

    # Print results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)
    print("\nBest Parameters Found:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest Cross-Validation F1 Score: {search.best_score_:.4f}")
    
    # Show top 5 configurations
    print("\nTop 5 Configurations:")
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    for idx, row in results_df.head(5).iterrows():
        print(f"  Rank {int(row['rank_test_score'])}: F1={row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    
    print("=" * 60)

    return search.best_estimator_
    

def evaluate_model(y_pred, y_test):
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        y_pred: Predicted labels
        y_test: True labels
    """
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    
    # Class names for better readability
    target_names = ['Ham', 'Spam', 'Phishing', 'Nigerian 419']
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("-" * 60)
    print("OVERALL METRICS")
    print("-" * 60)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall:    {recall_weighted:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")

    # Per-class metrics
    print("\n" + "-" * 60)
    print("PER-CLASS METRICS")
    print("-" * 60)
    
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    for i, class_name in enumerate(target_names):
        print(f"\n{class_name}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall:    {recall_per_class[i]:.4f}")
        print(f"  F1-Score:  {f1_per_class[i]:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print(cm)
    
    # Calculate per-class accuracy
    print("\n" + "-" * 60)
    print("PER-CLASS ACCURACY")
    print("-" * 60)
    for i, class_name in enumerate(target_names):
        class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"{class_name}: {class_accuracy:.4f} ({cm[i, i]}/{cm[i, :].sum()})")

    # Plot confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix Heatmap')
        
        # Save figure
        import os
        os.makedirs('artifacts', exist_ok=True)
        plt.savefig('artifacts/confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("\n✓ Confusion matrix saved to artifacts/confusion_matrix.png")
        
        # Show if in interactive mode
        plt.show()
    except Exception as e:
        print(f"\n⚠️ Could not generate confusion matrix plot: {e}")
    
    print("-" * 60)
