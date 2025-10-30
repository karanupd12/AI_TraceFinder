import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
import random
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern as sk_lbp
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pywt
import matplotlib
matplotlib.use("Agg")

# Page configuration
st.set_page_config(
    page_title="AI Tracefinder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OFFICIAL_DIR = os.path.join(DATA_DIR, "Official")
WIKI_DIR = os.path.join(DATA_DIR, "Wikipedia")
FLATFIELD_DIR = os.path.join(DATA_DIR, "Flatfield")
FEATURES_CSV = os.path.join(BASE_DIR, "metadata_features.csv")

# Model paths for baseline
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "svm.pkl")

# Model paths for CNN
CNN_MODEL_PATH = os.path.join(DATA_DIR, "scanner_hybrid_final.keras")
CNN_ENCODER_PATH = os.path.join(DATA_DIR, "hybrid_label_encoder.pkl")
CNN_SCALER_PATH = os.path.join(DATA_DIR, "hybrid_feat_scaler.pkl")
RES_PATH = os.path.join(DATA_DIR, "official_wiki_residuals.pkl")
FP_PATH = os.path.join(FLATFIELD_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(FLATFIELD_DIR, "fp_keys.npy")

# Extensions
EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
IMG_SIZE = (256, 256)

# Sidebar
st.sidebar.title("🔍 AI Tracefinder")
st.sidebar.markdown("---")
option = st.sidebar.selectbox(
    "Select Option",
    ["Data Visualization", "Evaluate Models", "Prediction"]
)

# ============================================
# UTILITY FUNCTIONS FOR BASELINE PREDICTION
# ============================================
def load_and_preprocess_baseline(img_path, size=(512, 512)):
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
    ent = entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6)
    
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

# ============================================
# UTILITY FUNCTIONS FOR CNN PREDICTION
# ============================================
def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K+1)
    return [float(mag[(r >= bins[i]) & (r < bins[i+1])].mean() if ((r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

def preprocess_residual_pywt(img_array):
    if img_array.ndim == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img = img_array
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    return (img - den).astype(np.float32)

# ============================================
# 1. DATA VISUALIZATION
# ============================================
if option == "Data Visualization":
    st.title("📊 Data Visualization & Exploratory Data Analysis")
    st.markdown("---")
    
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["Official", "Wikipedia", "Flatfield", "Features CSV"]
    )
    
    if dataset_choice == "Features CSV":
        st.subheader("📄 Features CSV Analysis")
        
        if os.path.exists(FEATURES_CSV):
            df = pd.read_csv(FEATURES_CSV)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Features", len(df.columns))
            with col3:
                st.metric("Classes", df['class_label'].nunique() if 'class_label' in df.columns else "N/A")
            
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(10))
            
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe())
            
            if 'class_label' in df.columns:
                st.subheader("📈 Class Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                class_counts = df['class_label'].value_counts()
                ax.bar(class_counts.index, class_counts.values, color='#FF6B35')
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                ax.set_title("Class Distribution")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            st.subheader("🔥 Feature Correlation Heatmap")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
        else:
            st.error("Features CSV file not found!")
    
    else:
        dataset_map = {
            "Official": OFFICIAL_DIR,
            "Wikipedia": WIKI_DIR,
            "Flatfield": FLATFIELD_DIR
        }
        dataset_path = dataset_map[dataset_choice]
        
        st.subheader(f"📁 {dataset_choice} Dataset Analysis")
        
        if os.path.exists(dataset_path):
            class_counts = {}
            image_shapes = []
            brightness_values = []
            
            for class_name in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                count = 0
                for root, dirs, files in os.walk(class_path):
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in EXTENSIONS:
                            file_path = os.path.join(root, f)
                            img = cv2.imread(file_path)
                            if img is not None:
                                h, w = img.shape[:2]
                                image_shapes.append((h, w))
                                brightness_values.append(img.mean())
                                count += 1
                
                class_counts[class_name] = count
            
            total_images = sum(class_counts.values())
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", total_images)
            with col2:
                st.metric("Number of Classes", len(class_counts))
            with col3:
                avg_per_class = total_images / len(class_counts) if class_counts else 0
                st.metric("Avg Images/Class", f"{avg_per_class:.1f}")
            
            st.subheader("📊 Class Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(class_counts.keys(), class_counts.values(), color='#FF6B35')
            ax.set_xlabel("Class")
            ax.set_ylabel("Number of Images")
            ax.set_title(f"Class Distribution - {dataset_choice}")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            if image_shapes:
                st.subheader("📐 Image Dimensions Analysis")
                heights, widths = zip(*image_shapes)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Height Statistics:**")
                    st.write(f"Min: {min(heights)} | Max: {max(heights)} | Mean: {np.mean(heights):.2f}")
                with col2:
                    st.write("**Width Statistics:**")
                    st.write(f"Min: {min(widths)} | Max: {max(widths)} | Mean: {np.mean(widths):.2f}")
                
                st.subheader("📏 Aspect Ratio Distribution")
                aspect_ratios = [w/h for h, w in image_shapes]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(aspect_ratios, bins=30, color='#4ECDC4', edgecolor='black')
                ax.set_xlabel("Width / Height")
                ax.set_ylabel("Frequency")
                ax.set_title("Aspect Ratio Distribution")
                st.pyplot(fig)
                
                st.subheader("💡 Brightness Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(brightness_values, bins=30, color='#95E1D3', edgecolor='black')
                ax.set_xlabel("Mean Pixel Intensity")
                ax.set_ylabel("Frequency")
                ax.set_title("Brightness Distribution")
                st.pyplot(fig)
            
            st.subheader("🖼️ Sample Images")
            for class_name in list(class_counts.keys())[:3]:
                class_path = os.path.join(dataset_path, class_name)
                all_images = []
                for root, dirs, files in os.walk(class_path):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in EXTENSIONS:
                            all_images.append(os.path.join(root, f))
                
                if all_images:
                    sample_files = random.sample(all_images, min(3, len(all_images)))
                    st.write(f"**{class_name}**")
                    cols = st.columns(3)
                    for i, fpath in enumerate(sample_files):
                        img = cv2.imread(fpath)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        with cols[i]:
                            st.image(img_rgb, caption=os.path.basename(fpath), use_container_width=True)
        else:
            st.error(f"Dataset path not found: {dataset_path}")

# ============================================
# 2. EVALUATE MODELS
# ============================================
elif option == "Evaluate Models":
    st.title("🎯 Model Evaluation")
    st.markdown("---")
    
    model_choice = st.selectbox("Select Model", ["Baseline (Random Forest & SVM)", "CNN Model"])
    
    if st.button("Evaluate Model"):
        with st.spinner(f"Evaluating {model_choice}..."):
            
            if model_choice == "Baseline (Random Forest & SVM)":
                try:
                    if not os.path.exists(FEATURES_CSV):
                        st.error(f"Features CSV not found at: {FEATURES_CSV}")
                    else:
                        df = pd.read_csv(FEATURES_CSV)
                        X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
                        y = df["class_label"]
                        
                        scaler = joblib.load(SCALER_PATH)
                        X_scaled = scaler.transform(X)
                        
                        st.subheader("🌲 Random Forest Evaluation")
                        rf_model = joblib.load(RF_MODEL_PATH)
                        y_pred_rf = rf_model.predict(X_scaled)
                        
                        st.text("Classification Report:")
                        st.text(classification_report(y, y_pred_rf))
                        
                        cm_rf = confusion_matrix(y, y_pred_rf, labels=rf_model.classes_)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm_rf, annot=True, fmt="d",
                                    xticklabels=rf_model.classes_,
                                    yticklabels=rf_model.classes_,
                                    cmap="Blues")
                        ax.set_title("Random Forest Confusion Matrix", fontsize=14)
                        ax.set_xlabel("Predicted", fontsize=12)
                        ax.set_ylabel("True", fontsize=12)
                        st.pyplot(fig)
                        
                        st.subheader("🔷 SVM Evaluation")
                        svm_model = joblib.load(SVM_MODEL_PATH)
                        y_pred_svm = svm_model.predict(X_scaled)
                        
                        st.text("Classification Report:")
                        st.text(classification_report(y, y_pred_svm))
                        
                        cm_svm = confusion_matrix(y, y_pred_svm, labels=svm_model.classes_)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm_svm, annot=True, fmt="d",
                                    xticklabels=svm_model.classes_,
                                    yticklabels=svm_model.classes_,
                                    cmap="Oranges")
                        ax.set_title("SVM Confusion Matrix", fontsize=14)
                        ax.set_xlabel("Predicted", fontsize=12)
                        ax.set_ylabel("True", fontsize=12)
                        st.pyplot(fig)
                        
                        st.success("✅ Baseline models evaluation complete!")
                        
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
            
            else:  # CNN Model
                try:
                    # Import TensorFlow only when needed
                    import tensorflow as tf
                    
                    with open(CNN_ENCODER_PATH, "rb") as f:
                        le = pickle.load(f)
                    with open(CNN_SCALER_PATH, "rb") as f:
                        scaler = pickle.load(f)
                    
                    model = tf.keras.models.load_model(CNN_MODEL_PATH)
                    
                    with open(RES_PATH, "rb") as f:
                        residuals_dict = pickle.load(f)
                    with open(FP_PATH, "rb") as f:
                        scanner_fps = pickle.load(f)
                    fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
                    
                    X_img_te, X_feat_te, y_te = [], [], []
                    for dataset_name in ["Official", "Wikipedia"]:
                        for scanner, dpi_dict in residuals_dict[dataset_name].items():
                            for dpi, res_list in dpi_dict.items():
                                for res in res_list:
                                    X_img_te.append(np.expand_dims(res, -1))
                                    v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
                                    v_fft = fft_radial_energy(res)
                                    v_lbp = lbp_hist_safe(res)
                                    X_feat_te.append(v_corr + v_fft + v_lbp)
                                    y_te.append(scanner)
                    
                    X_img_te = np.array(X_img_te, dtype=np.float32)
                    X_feat_te = np.array(X_feat_te, dtype=np.float32)
                    y_int_te = np.array([le.transform([c])[0] for c in y_te])
                    
                    X_feat_te = scaler.transform(X_feat_te)
                    
                    y_pred_prob = model.predict([X_img_te, X_feat_te])
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    
                    test_acc = accuracy_score(y_int_te, y_pred)
                    
                    st.subheader("🧠 CNN Model Evaluation")
                    st.metric("Test Accuracy", f"{test_acc*100:.2f}%")
                    
                    st.text("Classification Report:")
                    st.text(classification_report(y_int_te, y_pred, target_names=le.classes_))
                    
                    cm = confusion_matrix(y_int_te, y_pred)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=le.classes_, yticklabels=le.classes_)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title("CNN Confusion Matrix")
                    st.pyplot(fig)
                    
                    st.success("✅ CNN model evaluation complete!")
                    
                except Exception as e:
                    st.error(f"Error during CNN evaluation: {str(e)}")

# ============================================
# 3. PREDICTION
# ============================================
elif option == "Prediction":
    st.title("🔮 Scanner Prediction")
    st.markdown("---")
    
    model_choice = st.selectbox("Select Model for Prediction", ["Baseline (Random Forest)", "CNN Model"])
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Prediction Results")
            
            if st.button("Predict"):
                with st.spinner("Making prediction..."):
                    
                    if model_choice == "Baseline (Random Forest)":
                        try:
                            # Save uploaded file temporarily
                            temp_path = "temp_upload.tif"
                            image.save(temp_path)
                            
                            # Load scaler and model
                            scaler = joblib.load(SCALER_PATH)
                            model = joblib.load(RF_MODEL_PATH)
                            
                            # Process image
                            img = load_and_preprocess_baseline(temp_path)
                            features = compute_metadata_features(img, temp_path)
                            
                            # Predict
                            df = pd.DataFrame([features])
                            X_scaled = scaler.transform(df)
                            pred = model.predict(X_scaled)[0]
                            prob = model.predict_proba(X_scaled)[0]
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            
                            # Display results
                            st.success(f"✅ **Predicted Scanner: {pred}**")
                            
                            # Get top 3 predictions
                            top_indices = np.argsort(prob)[::-1][:3]
                            
                            st.subheader("📊 Top 3 Predictions")
                            for idx in top_indices:
                                scanner_name = model.classes_[idx]
                                confidence = prob[idx] * 100
                                st.metric(scanner_name, f"{confidence:.2f}%")
                            
                            # Show probability distribution
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sorted_indices = np.argsort(prob)[::-1]
                            sorted_classes = [model.classes_[i] for i in sorted_indices]
                            sorted_probs = [prob[i] * 100 for i in sorted_indices]
                            
                            ax.barh(sorted_classes, sorted_probs, color='#FF6B35')
                            ax.set_xlabel("Confidence (%)")
                            ax.set_title("Scanner Prediction Probabilities")
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                    
                    else:  # CNN Model
                        try:
                            # Import TensorFlow only when needed
                            import tensorflow as tf
                            
                            # Load model and preprocessors
                            model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
                            
                            with open(CNN_ENCODER_PATH, "rb") as f:
                                le_inf = pickle.load(f)
                            with open(CNN_SCALER_PATH, "rb") as f:
                                scaler_inf = pickle.load(f)
                            with open(FP_PATH, "rb") as f:
                                scanner_fps_inf = pickle.load(f)
                            fp_keys_inf = np.load(ORDER_NPY, allow_pickle=True).tolist()
                            
                            # Process image
                            img_array = np.array(image)
                            res = preprocess_residual_pywt(img_array)
                            
                            # Prepare inputs
                            x_img = np.expand_dims(res, axis=(0,-1))
                            v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
                            v_fft = fft_radial_energy(res)
                            v_lbp = lbp_hist_safe(res)
                            v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
                            x_feat = scaler_inf.transform(v)
                            
                            # Predict
                            prob = model.predict([x_img, x_feat], verbose=0)
                            idx = int(np.argmax(prob))
                            label = le_inf.classes_[idx]
                            conf = float(prob[0, idx]*100)
                            
                            # Display results
                            st.success(f"✅ **Predicted Scanner: {label}**")
                            st.metric("Confidence", f"{conf:.2f}%")
                            
                            # Get top 3 predictions
                            top_indices = np.argsort(prob[0])[::-1][:3]
                            
                            st.subheader("📊 Top 3 Predictions")
                            for idx in top_indices:
                                scanner_name = le_inf.classes_[idx]
                                confidence = prob[0][idx] * 100
                                st.metric(scanner_name, f"{confidence:.2f}%")
                            
                            # Show probability distribution
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sorted_indices = np.argsort(prob[0])[::-1]
                            sorted_classes = [le_inf.classes_[i] for i in sorted_indices]
                            sorted_probs = [prob[0][i] * 100 for i in sorted_indices]
                            
                            ax.barh(sorted_classes, sorted_probs, color='#4ECDC4')
                            ax.set_xlabel("Confidence (%)")
                            ax.set_title("Scanner Prediction Probabilities")
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error during CNN prediction: {str(e)}")

