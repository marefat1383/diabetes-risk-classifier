import argparse

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    data = load_diabetes()
    threshold = float(np.median(data.target))
    labels = np.where(data.target >= threshold, "higher-risk", "lower-risk")
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(x_train, y_train)
    return model, x_test, y_test


def predict_risk(index: int = 0) -> tuple[str, str]:
    model, x_test, y_test = build_model()
    if index < 0 or index >= len(x_test):
        raise ValueError(f"Index must be between 0 and {len(x_test) - 1}")
    return str(model.predict([x_test[index]])[0]), str(y_test[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an educational diabetes-risk class.")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    predicted, actual = predict_risk(args.index)
    print(f"Predicted: {predicted} | Actual: {actual}")


if __name__ == "__main__":
    main()
