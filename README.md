# Property Address Classifier

A machine learning classifier to categorize property addresses into predefined categories: flat, houseorplot, landparcel, commercial unit, and others.

## 📁 Project Structure

```
property-classifier/
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Processed datasets
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   └── 02_modeling.ipynb    # Model training and evaluation
├── src/
│   ├── train.py            # Training script
│   └── predict.py          # Prediction script
├── best_model/             # Saved model artifacts
│   ├── classifier.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
├── results/                # Evaluation results and plots
├── requirements.txt        # Python dependencies
├── approach.txt           # Detailed methodology
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Place Data

Put your datasets in the `data/raw/` folder:
- `train.csv`
- `validation.csv`

### 3. Train Model

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train an XGBoost classifier
- Evaluate on validation set
- Save the model to `best_model/`

### 4. Make Predictions

**Single prediction:**
```bash
python src/predict.py "Flat 101, Tower A, Green Valley Apartments"
```

**Interactive mode:**
```bash
python src/predict.py
```

## 📊 Model Performance

**Validation Results:**
- Accuracy: 89.78%
- Macro F1 Score: 0.8823
- Weighted F1 Score: 0.8988

See `results/` folder for detailed classification reports and confusion matrices.

## 🔧 Technical Details

**Features:**
- TF-IDF vectorization with n-grams (1-3)
- 5000 maximum features
- Min document frequency: 2

**Model:**
- XGBoost Classifier
- 200 estimators
- Max depth: 6
- Learning rate: 0.1

**Preprocessing:**
- Lowercase conversion
- Unicode normalization
- N-gram extraction (unigrams, bigrams, trigrams)

## 📝 Notebooks

Explore the notebooks for detailed analysis:
1. `01_eda.ipynb` - Data exploration and visualization
2. `02_modeling.ipynb` - Model comparison and selection

## 🤝 Author

Vipul Pawar

## 📄 License

This project is for educational/assignment purposes. 
