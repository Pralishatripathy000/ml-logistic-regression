import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer


def main():
    model = joblib.load("models/logistic_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")

    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame

    X = df.drop("target", axis=1)

    sample = X.iloc[[0]]
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0]

    class_name = "Benign" if prediction == 1 else "Malignant"

    print("Sample Features:")
    print(sample)

    print("\nPredicted Class:")
    print(class_name)

    print("\nPrediction Probabilities:")
    print(pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Probability": probability
    }))


if __name__ == "__main__":
    main()