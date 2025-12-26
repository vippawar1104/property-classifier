"""
Property Address Classifier - Training Script
Usage: python src/train.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load training and validation data"""
    print("Loading data...")
    train_df = pd.read_csv('data/raw/train.csv')
    val_df = pd.read_csv('data/raw/validation.csv')
    
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    
    return train_df, val_df


def prepare_features(train_df, val_df):
    """Create TF-IDF features"""
    print("\nPreparing features...")
    
    # Extract text and labels
    X_train = train_df['property_address']
    y_train = train_df['categories']
    X_val = val_df['property_address']
    y_val = val_df['categories']
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        lowercase=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    print(f"  Feature dimensions: {X_train_tfidf.shape}")
    print(f"  Number of classes: {len(label_encoder.classes_)}")
    
    return (X_train_tfidf, X_val_tfidf, 
            y_train_encoded, y_val_encoded, 
            y_val, vectorizer, label_encoder)


def train_model(X_train, y_train):
    """Train XGBoost classifier"""
    print("\nTraining XGBoost model...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    print("  Training complete!")
    
    return model


def evaluate_model(model, X_val, y_val_encoded, y_val_original, label_encoder):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    # Predictions
    y_pred_encoded = model.predict(X_val)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Metrics
    accuracy = accuracy_score(y_val_original, y_pred)
    macro_f1 = f1_score(y_val_original, y_pred, average='macro')
    weighted_f1 = f1_score(y_val_original, y_pred, average='weighted')
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\n" + classification_report(y_val_original, y_pred))
    
    return accuracy, macro_f1


def save_model(model, vectorizer, label_encoder):
    """Save model artifacts"""
    print("\nSaving model artifacts...")
    
    # Create directory if doesn't exist
    os.makedirs('best_model', exist_ok=True)
    
    # Save files
    joblib.dump(model, 'best_model/classifier.pkl')
    joblib.dump(vectorizer, 'best_model/vectorizer.pkl')
    joblib.dump(label_encoder, 'best_model/label_encoder.pkl')
    
    print("  ✅ Saved: best_model/classifier.pkl")
    print("  ✅ Saved: best_model/vectorizer.pkl")
    print("  ✅ Saved: best_model/label_encoder.pkl")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("PROPERTY ADDRESS CLASSIFIER - TRAINING")
    print("="*60 + "\n")
    
    # Load data
    train_df, val_df = load_data()
    
    # Prepare features
    (X_train, X_val, y_train_enc, y_val_enc, 
     y_val_orig, vectorizer, label_encoder) = prepare_features(train_df, val_df)
    
    # Train model
    model = train_model(X_train, y_train_enc)
    
    # Evaluate
    accuracy, macro_f1 = evaluate_model(
        model, X_val, y_val_enc, y_val_orig, label_encoder
    )
    
    # Save model
    save_model(model, vectorizer, label_encoder)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Validation Accuracy: {accuracy:.4f}")
    print(f"Final Validation Macro F1: {macro_f1:.4f}")
    print("\nModel ready for inference using src/predict.py")


if __name__ == "__main__":
    main()