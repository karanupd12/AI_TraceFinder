import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os

# Use non-GUI backend to avoid Tkinter error
matplotlib.use("Agg")

CSV_PATH = "C:/Users/karan/OneDrive/Desktop/AI_TraceFinder/metadata_features.csv"

def evaluate_model(model_path, name, save_dir="results"):
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
    y = df["class_label"]

    # Load scaler + model
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load(model_path)

    # Transform features
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Print classification report
    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y, y_pred))

    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        cmap="Blues"
    )
    plt.title(f"{name} Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)

    # Ensure results folder exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")

    # Save figure instead of showing
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved to: {save_path}")

if __name__ == "__main__":
    evaluate_model("models/random_forest.pkl", "Random Forest")
    evaluate_model("models/svm.pkl", "SVM")
