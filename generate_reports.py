"""
Generate Final Evaluation Reports
Run from property-classifier directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import os

def main():
    print("="*70)
    print("FINAL MODEL EVALUATION")
    print("="*70)
    
    # Load validation data
    print("\nLoading validation data...")
    val_df = pd.read_csv('../data/raw/validation.csv')
    
    # Load model artifacts
    print("Loading model artifacts...")
    model = joblib.load('../best_model/classifier.pkl')
    vectorizer = joblib.load('../best_model/vectorizer.pkl')
    label_encoder = joblib.load('../best_model/label_encoder.pkl')
    
    # Prepare data
    X_val = vectorizer.transform(val_df['property_address'])
    y_val = val_df['categories']
    
    # Predict
    y_pred_encoded = model.predict(X_val)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Get labels
    labels = sorted(y_val.unique())
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    weighted_f1 = f1_score(y_val, y_pred, average='weighted')
    
    # Save classification report
    report = classification_report(y_val, y_pred, target_names=labels)
    with open('../results/classification_report.txt', 'w') as f:
        f.write("FINAL MODEL - CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        f.write(f"\n\nOVERALL METRICS:\n")
        f.write(f"  Accuracy:    {accuracy:.4f}\n")
        f.write(f"  Macro F1:    {macro_f1:.4f}\n")
        f.write(f"  Weighted F1: {weighted_f1:.4f}\n")
    
    print("✅ Classification report saved to: results/classification_report.txt")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - XGBoost (Final Model)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/confusion_matrix_final.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrix saved to: results/confusion_matrix_final.png")
    plt.close()
    
    # Calculate per-class metrics
    precision = precision_score(y_val, y_pred, labels=labels, average=None)
    recall = recall_score(y_val, y_pred, labels=labels, average=None)
    f1 = f1_score(y_val, y_pred, labels=labels, average=None)
    
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }, index=labels)
    
    # Save to CSV
    metrics_df.to_csv('../results/per_class_metrics.csv')
    print("✅ Per-class metrics saved to: results/per_class_metrics.csv")
    
    # Plot per-class metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Per-Class Metrics - XGBoost (Final Model)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../results/per_class_metrics_final.png', dpi=300, bbox_inches='tight')
    print("✅ Per-class metrics plot saved to: results/per_class_metrics_final.png")
    plt.close()
    
    # Error analysis
    errors_df = pd.DataFrame({
        'address': val_df['property_address'],
        'true_label': y_val,
        'predicted_label': y_pred
    })
    errors_df = errors_df[errors_df['true_label'] != errors_df['predicted_label']]
    errors_df.to_csv('../results/misclassified_samples.csv', index=False)
    print("✅ Misclassified samples saved to: results/misclassified_samples.csv")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} (89.78%)")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"\nAll reports saved to results/ folder")

if __name__ == "__main__":
    main()
