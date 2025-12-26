"""
Evaluation Module for Property Address Classifier
This module has all the functions I need for evaluating and analyzing the model
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
    f1_score,
    cohen_kappa_score
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(y_true, y_pred, labels=None, model_name="Model"):
    """
    This function does a complete evaluation of the model with all the important metrics
    
    Args:
        y_true: The actual correct labels
        y_pred: What the model predicted
        labels: Names of the categories (like 'flat', 'commercial unit', etc.)
        model_name: Just a name to identify which model we're testing
        
    Returns:
        dict: A dictionary with all the performance metrics
    """
    print(f"{'='*70}")
    print(f"EVALUATION RESULTS: {model_name}")
    print(f"{'='*70}")
    
    # Let's print the detailed classification report first
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    # Now calculate all the important metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\nOVERALL METRICS:")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Macro Precision:   {macro_precision:.4f}")
    print(f"  Macro Recall:      {macro_recall:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Weighted F1:       {weighted_f1:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    
    # Pack everything into a dictionary to return
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'kappa': kappa
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None, model_name="Model", 
                         save_path=None, figsize=(10, 8)):
    """
    Creates a nice confusion matrix visualization
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        labels: Category names
        model_name: Model identifier for the title
        save_path: Where to save the plot (if needed)
        figsize: Size of the figure
    """
    # First, let's compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Make a nice heatmap with numbers showing
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the figure if a path was provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_per_class_metrics(y_true, y_pred, labels=None, model_name="Model",
                           save_path=None, figsize=(12, 6)):
    """
    Shows how well the model performs for each category separately
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Category names
        model_name: Model name for the title
        save_path: Where to save the plot
        figsize: Figure dimensions
    """
    # Calculate metrics for each category individually
    precision = precision_score(y_true, y_pred, labels=labels, average=None)
    recall = recall_score(y_true, y_pred, labels=labels, average=None)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None)
    
    # Put everything in a nice dataframe
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }, index=labels)
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=figsize)
    metrics_df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(f'Per-Class Metrics - {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save it if needed
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Per-class metrics plot saved to: {save_path}")
    
    plt.show()
    
    return metrics_df


def analyze_errors(y_true, y_pred, X_text, labels=None, top_n=20):
    """
    Takes a closer look at where the model got things wrong
    
    Args:
        y_true: Correct labels
        y_pred: What the model predicted
        X_text: The actual address texts
        labels: Category names
        top_n: How many error examples to show
        
    Returns:
        pd.DataFrame: All the misclassified samples
    """
    # Build a dataframe with everything
    errors_df = pd.DataFrame({
        'address': X_text,
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    # Keep only the ones where model was wrong
    errors_df = errors_df[errors_df['true_label'] != errors_df['predicted_label']]
    
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS")
    print(f"{'='*70}")
    print(f"\nTotal misclassifications: {len(errors_df)}")
    print(f"Error rate: {len(errors_df) / len(y_true) * 100:.2f}%")
    
    # Let's see which mistakes happen most often
    print(f"\nMOST COMMON MISCLASSIFICATION PATTERNS:")
    error_patterns = errors_df.groupby(['true_label', 'predicted_label']).size()
    error_patterns = error_patterns.sort_values(ascending=False).head(10)
    
    for (true, pred), count in error_patterns.items():
        print(f"  {true:20s} → {pred:20s}: {count:3d} errors")
    
    # Show some actual examples of mistakes
    print(f"\nSAMPLE MISCLASSIFICATIONS (top {min(top_n, len(errors_df))}):")
    print(f"{'-'*70}")
    
    for idx, row in errors_df.head(top_n).iterrows():
        print(f"\nAddress: {row['address']}")
        print(f"  True:      {row['true_label']}")
        print(f"  Predicted: {row['predicted_label']}")
    
    return errors_df


def compare_models(results_list, save_path=None, figsize=(14, 6)):
    """
    Compare multiple models side by side
    
    Args:
        results_list: List of metric dictionaries from evaluate_model()
        save_path: Path to save the comparison plot
        figsize: Figure size tuple
    """
    # Create dataframe
    results_df = pd.DataFrame(results_list)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy comparison
    results_df.plot(x='model_name', y='accuracy', kind='bar', 
                   ax=axes[0], color='steelblue', legend=False)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # F1 Score comparison
    results_df.plot(x='model_name', y='macro_f1', kind='bar', 
                   ax=axes[1], color='coral', legend=False)
    axes[1].set_title('Model Macro F1 Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Macro F1 Score', fontsize=12)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Model comparison plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
    return results_df


def evaluate_from_saved_model(model_path, vectorizer_path, encoder_path, 
                              val_df, text_column='property_address',
                              label_column='categories'):
    """
    Load saved model and evaluate on validation data
    
    Args:
        model_path: Path to saved model (.pkl)
        vectorizer_path: Path to saved vectorizer (.pkl)
        encoder_path: Path to saved label encoder (.pkl)
        val_df: Validation dataframe
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        dict: Evaluation metrics
    """
    print("Loading saved model...")
    
    # Load artifacts
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)
    
    print("  ✅ Model loaded successfully")
    
    # Prepare data
    X_val = vectorizer.transform(val_df[text_column])
    y_val = val_df[label_column]
    
    # Predict
    y_pred_encoded = model.predict(X_val)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Evaluate
    labels = sorted(y_val.unique())
    metrics = evaluate_model(y_val, y_pred, labels=labels, model_name="Saved Model")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, y_pred, labels=labels, 
                         model_name="Saved Model",
                         save_path='results/confusion_matrix_saved_model.png')
    
    # Plot per-class metrics
    plot_per_class_metrics(y_val, y_pred, labels=labels,
                          model_name="Saved Model",
                          save_path='results/per_class_metrics_saved_model.png')
    
    # Analyze errors
    analyze_errors(y_val, y_pred, val_df[text_column], labels=labels)
    
    return metrics


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - evaluate_model()")
    print("  - plot_confusion_matrix()")
    print("  - plot_per_class_metrics()")
    print("  - analyze_errors()")
    print("  - compare_models()")
    print("  - evaluate_from_saved_model()")