# 📊 AI TraceFinder: Forensic Scanner Identification

Detecting document forgery by analyzing a scanner's unique digital fingerprint.

---

## 🎯 Overview

Scanned documents like legal agreements, official certificates, and financial records are easy to forge. It's often impossible to tell if a scanned document is legitimate or if it was created using an unauthorized, fraudulent device.

**AI TraceFinder** is a forensic machine learning platform that identifies the source scanner device used to digitize a document or image. Every scanner, due to its unique hardware, introduces microscopic and invisible "fingerprints" into an image. These include specific noise patterns, texture artifacts, and compression traces. This project uses machine learning to train models that recognize these unique signatures, enabling fraud detection, authentication, and forensic validation in scanned documents.

By analyzing these patterns, this project allows you to:
- Attribute a scanned document to a specific scanner model  
- Detect forgeries where unauthorized scanners were used  
- Verify the authenticity of scanned evidence in a forensic context  
- Differentiate between authentic vs. tampered scans  

---

## 🛠 Tech Stack

This project leverages a modern stack for machine learning, image processing, and web application delivery.

| Category | Technology | Purpose |
|-----------|-------------|----------|
| **Backend & ML** | **Python** | Core programming language |
| | **Scikit-learn** | Random Forest & SVM (Baseline Models) |
| | **Pandas** | Data manipulation and CSV handling |
| | **OpenCV** | Image processing (loading, color conversion, etc.) |
| | **NumPy** | Numerical operations |
| | **TensorFlow / Keras** | For CNN Model |
| **Frontend & UI** | **Streamlit** | Creating the interactive web application |
| | **Matplotlib & Seaborn** | Data visualization (confusion matrix, plots) |
| | **Pillow (PIL)** | Displaying sample images in the UI |
| **Tooling** | **Git & GitHub** | Version control and source management |
| | **venv** | Python virtual environment management |

---

## ✨ Features

- 🧩 **Modular Feature Extraction:** Streamlit app to scan image directories, extract 10+ metadata features, and generate a feature CSV 
- 📊 **Data Visualization:** View class distribution graphs, sample images from each class, and a full data preview  
- 💾 **Downloadable Results:** Download the complete feature CSV directly from the app  
- 🤖 **Baseline Model Pipeline:** 
  - **Train:** Build Random Forest and SVM models from the feature CSV  
  - **Evaluate:** View detailed classification reports and confusion matrices for both models  
  - **Predict:** Upload any image for instant scanner identification   
- 🧠 **Deep Learning Model:** Integration of a CNN for end-to-end image-based classification

---

## 📂 Dataset

This project uses the [NIST OpenMFC](https://www.nist.gov/) scanner dataset containing scans from multiple device models at various DPI settings (150, 300, 600).

---

## 📸 Architecture & UI

### System Architecture
![System Architecture](./images/Architecture.png)

### Overview
![Overview](./images/Overview.png)

### Exploratory Data Analysis
![EDA](./images/EDA.png)

### Model Evaluation
![Evaluation](./images/Evaluation.png)

### Prediction Interface
![Prediction](./images/Prediction.png)

---

## 📁 Suggested Project Structure

```
tracefinderPred/
├── Data/                         # Raw dataset
├── models/                       # Trained ML models
├── pre_process/                  # Data preprocessing scripts
├── processed_data/               # Cleaned and processed datasets
├── results/                      # Model evaluation results
├── scr/(Baseline and CNN)        # Source code modules
├── venv/                         # Virtual environment
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── LICENSE                       # Project license
└── Readme.md                     # Project documentation
```

## 📈 Accuracy & Performance

- **Hybrid CNN model model accuracy**: 85%
- **Overall weighted avg:** Precision: 0.858, Recall: 0.852, F1-score: 0.852
- **Test sample:** 500 images
- **Average Test Confidence: 93.85%**

---

## 💼 Applications

### Digital Forensics
**Description:** Determine which scanner was used to forge or duplicate legal documents.  
**Example:** Detect whether a fake certificate was created using a specific scanner model.

### Document Authentication
**Description:** Identify the source of printed and scanned images to detect tampering or fraudulent claims.  
**Example:** Differentiate between scans from authorized and unauthorized departments.

### Legal Evidence Verification
**Description:** Ensure scanned copies submitted in court/legal matters came from known and approved devices.  
**Example:** Verify that scanned agreements originated from the company's official scanner.

---

---

## 📋 Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Virtual environment tool (venv or virtualenv)

---

## 🚀 Project Setup

Follow these steps to set up the project locally.

### 1️⃣ Clone the Repository
```
git clone https://github.com/karanupadhyay1/AI_TraceFinder.git
cd tracefinderPred
```

### 2️⃣ Create and Activate Virtual Environment
**For Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit Application
```
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

## 📧 Contact

**Karan Upadhyay**

- **Email:** karanupadhyay@example.com
- **LinkedIn:** [linkedin.com/in/karanupadhyay](https://linkedin.com/in/karanupadhyay)
- **GitHub:** [github.com/karanupadhyay1](https://github.com/karanupadhyay1)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Special thanks to all contributors and the open-source community for their invaluable resources and support. This project was developed as part of forensic research to combat document forgery and enhance digital authentication systems.

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**