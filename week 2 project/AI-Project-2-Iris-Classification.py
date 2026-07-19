"""
=====================================================================
Project      : Iris Flower Classification using K-Nearest Neighbors
Internship   : DecodeLabs AI - Project 2
Author       : <Your Name>
=====================================================================

Description
-----------
An end-to-end, production-style ML pipeline for the classic Iris
dataset. Covers data loading, EDA, preprocessing, hyperparameter
tuning (GridSearchCV + cross-validation), model evaluation, rich
visualizations, and model persistence (so the trained model can be
reloaded later without retraining).

Run
---
    python iris_knn_classifier.py

Outputs (written to ./outputs/)
--------------------------------
    iris.csv                    - raw dataset export
    species_count.png           - class balance bar chart
    feature_pairplot.png        - pairwise feature relationships
    correlation_heatmap.png     - feature correlation heatmap
    k_accuracy_curve.png        - accuracy vs. k (model selection)
    confusion_matrix.png        - heatmap of the confusion matrix
    iris_knn_model.joblib       - trained model + scaler bundle
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
K_RANGE = range(1, 21)
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("iris-knn")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the Iris dataset into a labeled DataFrame and export it as CSV."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[i] for i in iris.target]
    df.to_csv(OUTPUT_DIR / "iris.csv", index=False)
    log.info("Dataset loaded: %d rows x %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
def explore_data(df: pd.DataFrame) -> None:
    """Print summary statistics and save exploratory plots."""
    print("=" * 60)
    print("First 5 Rows")
    print("=" * 60)
    print(df.head(), "\n")

    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns), "\n")

    print("=" * 60)
    print("Dataset Info")
    print("=" * 60)
    df.info()
    print()

    print("=" * 60)
    print("Statistical Summary")
    print("=" * 60)
    print(df.describe(), "\n")

    print("=" * 60)
    print("Missing Values Per Column")
    print("=" * 60)
    print(df.isnull().sum(), "\n")

    print("=" * 60)
    print("Class Balance")
    print("=" * 60)
    print(df["species"].value_counts(), "\n")

    # --- Plot: class balance ---
    plt.figure(figsize=(6, 5))
    df["species"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"])
    plt.title("Number of Flowers per Species")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "species_count.png", dpi=150)
    plt.show()
    plt.close()

    # --- Plot: pairwise feature relationships ---
    pairplot = sns.pairplot(df, hue="species", diag_kind="kde", palette="Set2")
    pairplot.fig.suptitle("Pairwise Feature Relationships by Species", y=1.02)
    pairplot.savefig(OUTPUT_DIR / "feature_pairplot.png", dpi=150)
    plt.show()
    plt.close("all")
    # --- Plot: correlation heatmap ---
    plt.figure(figsize=(6, 5))
    numeric_df = df.drop(columns="species")
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.show()
    plt.close()

    log.info("EDA plots saved to '%s/'", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def split_and_scale(df: pd.DataFrame):
    """Split into train/test sets and standardize features."""
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # KNN is distance-based, so feature scaling matters a lot.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log.info(
        "Split complete: %d train rows, %d test rows (stratified)",
        len(X_train), len(X_test),
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


# ---------------------------------------------------------------------------
# 4. Model selection: find the best k via cross-validation
# ---------------------------------------------------------------------------
def find_best_k(X_train, y_train) -> int:
    """Sweep k values with cross-validation and plot accuracy vs. k."""
    cv_scores = []
    for k in K_RANGE:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
        cv_scores.append(scores.mean())

    plt.figure(figsize=(7, 5))
    plt.plot(list(K_RANGE), cv_scores, marker="o", color="#4C72B0")
    plt.title(f"{CV_FOLDS}-Fold CV Accuracy vs. k")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Mean CV Accuracy")
    plt.xticks(list(K_RANGE))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "k_accuracy_curve.png", dpi=150)
    plt.show()
    plt.close()
    best_k = list(K_RANGE)[int(np.argmax(cv_scores))]
    log.info("Best k found via cross-validation: k=%d (CV accuracy=%.4f)", best_k, max(cv_scores))
    return best_k


def tune_hyperparameters(X_train, y_train) -> KNeighborsClassifier:
    """Run GridSearchCV over k, weighting scheme, and distance metric."""
    param_grid = {
        "n_neighbors": list(K_RANGE),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    log.info("Best hyperparameters: %s", grid.best_params_)
    log.info("Best CV accuracy: %.4f", grid.best_score_)
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> None:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("Test Accuracy")
    print("=" * 60)
    print(f"{accuracy * 100:.2f}%\n")

    print("=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_test, predictions))

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", colorbar=False, ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.show()
    plt.close()

    log.info("Evaluation complete. Confusion matrix saved.")


# ---------------------------------------------------------------------------
# 6. Persistence + inference on a new sample
# ---------------------------------------------------------------------------
def save_model(model, scaler, feature_names) -> Path:
    bundle = {"model": model, "scaler": scaler, "feature_names": feature_names}
    path = OUTPUT_DIR / "iris_knn_model.joblib"
    joblib.dump(bundle, path)
    log.info("Model bundle saved to '%s'", path)
    return path


def predict_new_sample(model_path: Path, sample: list) -> str:
    bundle = joblib.load(model_path)
    sample_df = pd.DataFrame([sample], columns=bundle["feature_names"])
    scaled_sample = bundle["scaler"].transform(sample_df)
    prediction = bundle["model"].predict(scaled_sample)[0]
    return prediction


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    explore_data(df)

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    find_best_k(X_train, y_train)               # informative plot
    best_model = tune_hyperparameters(X_train, y_train)  # actual model used

    evaluate_model(best_model, X_test, y_test)

    model_path = save_model(best_model, scaler, feature_names)

    sample = [5.1, 3.5, 1.4, 0.2]  # classic setosa-like measurements
    result = predict_new_sample(model_path, sample)
    print("\n" + "=" * 60)
    print("Prediction for a new sample")
    print("=" * 60)
    print(f"Input  : {sample}")
    print(f"Species: {result}")


if __name__ == "__main__":
    main()