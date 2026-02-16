# Student Dropout and Academic Success Prediction

Machine learning project for predicting student outcomes in higher education using a 3-class classification approach: **Dropout**, **Enrolled**, and **Graduate**.

## Dataset

**Source**: [UCI Machine Learning Repository - Student Dropout Prediction](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

**Download instructions**:
1. Visit the dataset page linked above
2. Download the dataset (CSV format, semicolon-delimited)
3. Save it as `data 2.csv` in the project root directory and rename it to `data.csv`

## Setup

**Prerequisites**: Python 3.8+

**1. Create and activate virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**:
```bash
pip install -r requirements.txt
```

## Recreating the Results

**Run the analysis notebook**:
```bash
jupyter notebook notebook.ipynb
```

The notebook contains the complete analysis pipeline:
- Exploratory data analysis
- Feature engineering and selection
- Handling class imbalance (SMOTE)
- Model training (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Model evaluation and comparison
- Feature importance analysis
- Interpretability (SHAP, LIME)
- Fairness analysis

## Results

Generated figures from the first run has been saved in the `figures/` directory:
- Model performance comparisons
- Confusion matrices
- Feature importance plots
- SMOTE impact analysis
- Engineered features analysis

## Key Findings

The analysis uses an 80/20 train-test split and addresses class imbalance through SMOTE. Multiple classification algorithms are compared using metrics appropriate for imbalanced datasets (precision, recall, F1-score, ROC-AUC).
