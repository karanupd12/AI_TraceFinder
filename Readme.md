# AI TraceFinder — Forensic Scanner Identification

A machine learning platform that identifies the source scanner device used to digitize documents through analysis of unique noise patterns and artifacts.

---

## Overview

AI TraceFinder uses hybrid CNN architecture combined with handcrafted features (PRNU, FFT, LBP) to classify scanned documents across 11 scanner models with 93.65% accuracy.

---

## Features

- Scanner identification from scanned images
- Multi-model support (Hybrid CNN, Random Forest, SVM)
- Interactive Streamlit web interface
- Batch processing capabilities
- Real-time prediction with confidence scores

---

## Performance

**Test Accuracy: 93.65%**

- Training set: 3,654 scans
- Test set: 914 scans
- Scanner classes: 11 models (Canon, Epson, HP)
- Image resolution: 256×256 grayscale
- Features: 27 handcrafted + CNN-learned patterns

| Scanner Model | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Canon120-1 | 0.91 | 0.89 | 0.90 |
| Canon120-2 | 0.84 | 0.83 | 0.84 |
| Canon220 | 0.89 | 0.92 | 0.90 |
| Canon9000-1 | 0.95 | 0.87 | 0.91 |
| Canon9000-2 | 0.88 | 0.95 | 0.91 |
| EpsonV370-1 | 0.99 | 0.95 | 0.97 |
| EpsonV370-2 | 0.95 | 0.99 | 0.97 |
| EpsonV39-1 | 0.94 | 0.96 | 0.95 |
| EpsonV39-2 | 0.96 | 0.95 | 0.96 |
| EpsonV550 | 1.00 | 0.99 | 0.99 |
| HP | 0.99 | 1.00 | 0.99 |

---

## Installation

### Prerequisites
- Python 3.10.11
- 8GB+ RAM recommended

### Setup

1. Clone the repository
```
git clone https://github.com/karanupd12/AI-TraceFinder.git
cd AI-TraceFinder
```

2. Create virtual environment
```
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
```

3. Install dependencies
```
pip install -r Requirements.txt
```

---

## Usage

Run the Streamlit application:

```
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Features Available:
- **EDA**: Dataset statistics and visualizations
- **Prediction**: Upload images for scanner identification
- **Testing**: Batch processing of multiple images
- **Model Performance**: View classification metrics

---

## Dataset

This project uses the [NIST OpenMFC](https://www.nist.gov/) scanner dataset containing scans from multiple device models at various DPI settings (150, 300, 600).

---

## Project Structure

```
AI-TraceFinder/
├── streamlit_app.py          # Main application
├── Data/                     # Dataset
├── models/                   # Trained models
├── results/                  # Performance metrics
└── Requirements.txt          # Dependencies
```

---

## Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **ML**: scikit-learn (SVM, Random Forest)
- **Frontend**: Streamlit
- **Processing**: OpenCV, PyWavelets, NumPy

---

## License

MIT License

---

## Contact

**Karan Upadhyay**
- Email: karanupd12@gmail.com
- GitHub: [karanupd12](https://github.com/karanupd12)
- LinkedIn: [karanupd12](https://www.linkedin.com/in/karanupd12/)

---

## Acknowledgments

- NIST OpenMFC for the dataset
- TensorFlow and scikit-learn communities
```