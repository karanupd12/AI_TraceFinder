import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.stats import skew, kurtosis, entropy

st.set_page_config(page_title="Feature Extractor", layout="wide")
st.title("📷 Image feature extractor")

# Sidebar
dataset_root = st.sidebar.text_input("Dataset Path:", placeholder="path/to/dataset")
FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

def extract_features(image_path, camera):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"file_name": os.path.basename(image_path), "camera": camera, "error": "Unreadable"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024
        
        # Basic stats
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        
        b_mean, g_mean, r_mean = np.mean(img, axis=(0, 1))
        
        # Texture
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)
        
        # EXIF
        try:
            pil_img = Image.open(image_path)
            exif = pil_img._getexif() or {}
            camera_make = exif.get(271, 'Unknown')  # Make
            camera_model = exif.get(272, 'Unknown')  # Model
        except:
            camera_make = camera_model = 'Unknown'
        
        return {
            "file_name": os.path.basename(image_path),
            "camera": camera,
            "width": width,
            "height": height,
            "resolution_mp": round((width * height) / 1_000_000, 2),
            "file_size_kb": round(file_size, 2),
            "aspect_ratio": round(width / height, 3),
            "mean_intensity": round(mean_intensity, 3),
            "std_intensity": round(std_intensity, 3),
            "r_mean": round(r_mean, 3),
            "g_mean": round(g_mean, 3),
            "b_mean": round(b_mean, 3),
            "edge_density": round(edge_density, 3),
            "entropy": round(shannon_entropy, 3),
            "skewness": round(skew(gray.flatten()), 3),
            "kurtosis": round(kurtosis(gray.flatten()), 3)
        }
    except Exception as e:
        return {"file_name": os.path.basename(image_path), "camera": camera, "error": str(e)}


if dataset_root and os.path.isdir(dataset_root):
    camera_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    
    if not camera_folders:
        st.error("No camera folders found!")
        st.stop()
    
    selected_cameras = st.sidebar.multiselect("Select Cameras:", camera_folders, default=camera_folders)
    
    if st.sidebar.button("Extract Features"):
        if not selected_cameras:
            st.error("Select at least one camera!")
            st.stop()
        
        records = []
        total_images = sum(len([f for f in os.listdir(os.path.join(dataset_root, cam)) 
                               if f.lower().endswith(FORMATS)]) for cam in selected_cameras)
        
        progress = st.progress(0)
        processed = 0
        
        for camera in selected_cameras:
            camera_path = os.path.join(dataset_root, camera)
            images = [f for f in os.listdir(camera_path) if f.lower().endswith(FORMATS)]
            
            for image_file in images:
                image_path = os.path.join(camera_path, image_file)
                features = extract_features(image_path, camera)
                records.append(features)
                
                processed += 1
                progress.progress(processed / total_images)
        
        progress.empty()
        
        if records:
            df = pd.DataFrame(records)
            valid_df = df[~df.get('error').notna()].copy() if 'error' in df.columns else df.copy()
            
            st.success(f"Processed {len(valid_df)} images successfully")
            
            
            search = st.text_input("Search:", placeholder="filename or camera...")
            if search:
                mask = (valid_df['file_name'].str.contains(search, case=False, na=False) |
                       valid_df['camera'].str.contains(search, case=False, na=False))
                valid_df = valid_df[mask]
            
            # Table
            st.dataframe(valid_df, use_container_width=True)
            
            
            csv = valid_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "features.csv", "text/csv")
            
            
            save_path = os.path.join(dataset_root, "features.csv")
            valid_df.to_csv(save_path, index=False)
            st.info(f"Saved to: {save_path}")

elif dataset_root:
    st.error("Invalid path")
else:
    st.info("Enter dataset path to start")
