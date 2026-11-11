# Diabetes Risk Classifier

A small educational AI project that predicts a diabetes-risk class from health measurements using Logistic Regression and scikit-learn's built-in diabetes dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python diabetes_classifier.py --index 0
```

The index selects an example from the test set and prints the predicted and actual risk class.

## Test

```bash
python -m unittest test_diabetes_classifier.py
```

This is an educational demonstration, not a medical tool. Do not use it for diagnosis or treatment decisions.
