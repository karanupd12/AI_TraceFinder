import cv2
import numpy as np
import os
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.stats import skew, kurtosis, entropy


def extract_exif_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data = {}
        
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
        
        return exif_data
    except Exception:
        return {}


def extract_color_features(img):
    """Extract color-based features"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Color channel statistics
    b_mean, g_mean, r_mean = np.mean(img, axis=(0, 1))
    b_std, g_std, r_std = np.std(img, axis=(0, 1))
    
    # HSV statistics
    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
    
    # Color diversity (unique colors)
    unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
    
    return {
        "rgb_mean": (r_mean, g_mean, b_mean),
        "rgb_std": (r_std, g_std, b_std),
        "hsv_mean": (h_mean, s_mean, v_mean),
        "unique_colors": unique_colors,
        "color_diversity": unique_colors / (img.shape[0] * img.shape[1])
    }


def extract_texture_features(gray):
    """Extract texture-based features"""
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    
    # Gradient features
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Local Binary Pattern approximation
    lbp_var = np.var(gray)
    
    return {
        "edge_density": edge_density,
        "gradient_mean": np.mean(gradient_magnitude),
        "gradient_std": np.std(gradient_magnitude),
        "texture_variance": lbp_var
    }


def extract_features(image_path, camera_name):
    """Main feature extraction function"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {"file_name": os.path.basename(image_path), "camera": camera_name, "error": "Unreadable file"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Basic properties
        height, width, channels = img.shape
        file_size = os.path.getsize(image_path) / 1024  # KB
        aspect_ratio = round(width / height, 3)
        
        # Statistical features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())
        
        # Histogram and entropy
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)
        
        # Color features
        color_features = extract_color_features(img)
        
        # Texture features
        texture_features = extract_texture_features(gray)
        
        # EXIF data
        exif_data = extract_exif_data(image_path)
        camera_make = exif_data.get('Make', 'Unknown')
        camera_model = exif_data.get('Model', 'Unknown')
        iso = exif_data.get('ISOSpeedRatings', 'Unknown')
        
        # Noise estimation (using Laplacian variance)
        noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Compression artifacts (JPEG quality estimation)
        quality_score = estimate_jpeg_quality(gray)
        
        return {
            "file_name": os.path.basename(image_path),
            "camera": camera_name,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "width": width,
            "height": height,
            "channels": channels,
            "resolution_mp": round((width * height) / 1_000_000, 2),
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "iso": iso,
            
            # Intensity statistics
            "mean_intensity": round(mean_intensity, 3),
            "entropy": round(shannon_entropy, 3),
            
            # Color features
            "r_mean": round(color_features["rgb_mean"][0], 3),
            "g_mean": round(color_features["rgb_mean"][1], 3),
            "b_mean": round(color_features["rgb_mean"][2], 3),
            "color_diversity": round(color_features["color_diversity"], 6),
            
            # Texture features
            "edge_density": round(texture_features["edge_density"], 3),

            
            # Quality metrics
            "noise_level": round(noise_level, 3),
            "quality_score": round(quality_score, 3)
        }
        
    except Exception as e:
        return {"file_name": os.path.basename(image_path), "camera": camera_name, "error": str(e)}


def estimate_jpeg_quality(gray):
    """Estimate JPEG quality based on DCT coefficients"""
    try:
        # Simple quality estimation based on image smoothness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-100 scale (higher = better quality)
        quality = min(100, max(0, 100 - (laplacian_var / 1000) * 10))
        return quality
    except:
        return 50  # Default quality score
