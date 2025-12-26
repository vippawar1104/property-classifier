"""
Property Address Classifier - Prediction Script
Usage: 
    python src/predict.py "123 Apartment Complex, Tower A"
    python src/predict.py
"""

import sys
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_model():
    """Load trained model and artifacts"""
    try:
        model = joblib.load('best_model/classifier.pkl')
        vectorizer = joblib.load('best_model/vectorizer.pkl')
        label_encoder = joblib.load('best_model/label_encoder.pkl')
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        print("❌ Error: Model files not found!")
        print("   Please run 'python src/train.py' first to train the model.")
        sys.exit(1)


def predict_single(address, model, vectorizer, label_encoder):
    """Predict category for a single address"""
    # Vectorize
    X = vectorizer.transform([address])
    
    # Predict
    pred_encoded = model.predict(X)[0]
    prediction = label_encoder.inverse_transform([pred_encoded])[0]
    
    # Get probabilities (if available)
    try:
        proba = model.predict_proba(X)[0]
        confidence = max(proba) * 100
        
        # Get all class probabilities
        class_probs = dict(zip(label_encoder.classes_, proba))
        class_probs = {k: v*100 for k, v in class_probs.items()}
        class_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))
    except:
        confidence = None
        class_probs = None
    
    return prediction, confidence, class_probs


def interactive_mode():
    """Interactive prediction mode"""
    print("\n" + "="*60)
    print("PROPERTY ADDRESS CLASSIFIER - INTERACTIVE MODE")
    print("="*60)
    print("Type 'quit' or 'exit' to stop\n")
    
    model, vectorizer, label_encoder = load_model()
    
    while True:
        address = input("Enter property address: ").strip()
        
        if address.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not address:
            print("⚠️  Please enter a valid address\n")
            continue
        
        prediction, confidence, class_probs = predict_single(
            address, model, vectorizer, label_encoder
        )
        
        print(f"\n{'─'*60}")
        print(f"Address:    {address}")
        print(f"Category:   {prediction.upper()}")
        if confidence:
            print(f"Confidence: {confidence:.2f}%")
            print(f"\nAll probabilities:")
            for cat, prob in class_probs.items():
                bar = '█' * int(prob/2)
                print(f"  {cat:20s} {prob:5.2f}% {bar}")
        print(f"{'─'*60}\n")


def main():
    """Main prediction function"""
    
    # Check if address provided as argument
    if len(sys.argv) > 1:
        # Single prediction mode
        address = ' '.join(sys.argv[1:])
        
        print("\n" + "="*60)
        print("PROPERTY ADDRESS CLASSIFIER")
        print("="*60)
        
        model, vectorizer, label_encoder = load_model()
        prediction, confidence, class_probs = predict_single(
            address, model, vectorizer, label_encoder
        )
        
        print(f"\nAddress:    {address}")
        print(f"Category:   {prediction.upper()}")
        if confidence:
            print(f"Confidence: {confidence:.2f}%")
            print(f"\nAll probabilities:")
            for cat, prob in class_probs.items():
                bar = '█' * int(prob/2)
                print(f"  {cat:20s} {prob:5.2f}% {bar}")
        print("\n" + "="*60 + "\n")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()