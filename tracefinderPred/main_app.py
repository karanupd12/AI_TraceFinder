import streamlit as st
import pandas as pd
import joblib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

CSV_PATH = "C:/Users/karan/OneDrive/Desktop/AI_TraceFinder/metadata_features.csv"

# === TRAINING ===
def train_models():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Drop non-feature columns
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
    y = df["class_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/random_forest.pkl")

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, "models/svm.pkl")

    joblib.dump(scaler, "models/scaler.pkl")
    st.success("✅ Models trained and saved successfully!")

# === EVALUATION ===
def evaluate_model(model_path, name):
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
    y = df["class_label"]

    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load(model_path)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    st.subheader(f"{name} Classification Report")
    st.text(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} Confusion Matrix")
    st.pyplot(fig)

# === IMAGE PREPROCESSING ===
def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compute_metadata_features(img, file_path):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

# === PREDICTION ===
def predict_scanner(img_path, model_choice="rf"):
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load(f"models/{'random_forest' if model_choice == 'rf' else 'svm'}.pkl")

    img = load_and_preprocess(img_path)
    features = compute_metadata_features(img, img_path)

    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    return pred, prob

# === FEATURE EXPLORER ===
def feature_explorer():
    df = pd.read_csv(CSV_PATH)
    st.subheader("📊 Feature Explorer")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        feature = st.selectbox("Choose feature to visualize", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

# === STREAMLIT UI ===
st.title("📊 Scanner Classification App")

menu = st.sidebar.radio("Choose Action",
                        ["Feature Explorer", "Train Models", "Evaluate Models", "Predict Scanner"])

if menu == "Feature Explorer":
    feature_explorer()

elif menu == "Train Models":
    if st.button("Train Now"):
        train_models()

elif menu == "Evaluate Models":
    st.subheader("Evaluate Random Forest")
    evaluate_model("models/random_forest.pkl", "Random Forest")
    st.subheader("Evaluate SVM")
    evaluate_model("models/svm.pkl", "SVM")

elif menu == "Predict Scanner":
    uploaded_file = st.file_uploader("Upload a TIFF Image", type=["tif", "tiff"])
    model_choice = st.selectbox("Choose Model", ["Random Forest", "SVM"])
    if uploaded_file is not None:
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        try:
            pred, prob = predict_scanner(
                temp_path,
                model_choice="rf" if model_choice == "Random Forest" else "svm"
            )
            st.success(f"🖼 Predicted Scanner: {pred}")
            st.write("🔍 Class Probabilities:")
            model_file = f"models/{'random_forest' if model_choice == 'Random Forest' else 'svm'}.pkl"
            for cls, p in zip(joblib.load(model_file).classes_, prob):
                st.write(f"{cls}: {p:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")