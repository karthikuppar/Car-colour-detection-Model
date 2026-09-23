# Car Colour Detection & Traffic Signal Analytics Model

An advanced Computer Vision and Deep Learning system for real-time **vehicle detection**, **pedestrian counting**, and **multi-space car colour classification** (CIE-LAB + HSV + RGB Chromaticity).

---

## 🌟 Key Features

- **Object Detection (YOLOv8s):** Detects vehicles (cars, buses, trucks, motorcycles) and pedestrians across dense urban traffic scenes.
- **Precision Blue Car Classifier:** Uses multi-space chromaticity analysis (CIE-LAB $b^*$ shift + HSV hue/saturation + RGB dominance index) to accurately classify blue vehicles while rejecting silver/white/grey sky reflections, gold/warm paint, and blue clothing.
- **Multi-Region Spatial Sub-Quadrant Scanning:** Scans 6 spatial regions per bounding box to reliably detect blue vehicles even when partially occluded by other cars or pedestrians.
- **FastAPI REST Service:** High-performance async backend exposing `/api/analyze` and `/health` endpoints.
- **Modern React Dashboard (Vite + Glassmorphism):** Responsive UI featuring real-time visual canvas, drag-and-drop image upload, instant metric cards, and annotated image downloads.

---

## 🎨 Visual Bounding Box Standards

| Object Category | Bounding Box Color | Label Tag |
|---|---|---|
| **Blue Vehicles** | **RED** `(0, 0, 255)` | `Blue Car (Conf)` |
| **Other Vehicles** | **BLUE** `(255, 0, 0)` | `Other Car` |
| **Pedestrians** | **GREEN** `(0, 255, 0)` | `Person` |

---

## 📁 Project Architecture

```text
car/
├── backend/
│   ├── api.py                          # FastAPI server endpoints (/api/analyze, /health)
│   └── src/
│       ├── classification/
│       │   ├── __init__.py
│       │   └── color_detector.py       # Multi-Space CIE-LAB + HSV + RGB Color Classifier
│       ├── detection/
│       │   ├── __init__.py
│       │   └── detector.py             # YOLOv8 Object Detection Module
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── crop.py                 # Vehicle Crop Extraction Utility
│       └── utils/
│           ├── __init__.py
│           └── draw.py                 # Visual Annotator & Dashboard Pipeline
├── dataset/                            # Traffic images & vehicle crops
│   ├── crops_image/                    # Standalone car crops for color evaluation
│   ├── final_annotated_image.jpg       # Annotated visualization output
│   └── image.png                       # Primary test image
├── frontend/                           # React + Vite Glassmorphism Dashboard
│   ├── src/
│   │   ├── App.jsx                     # Main Dashboard Component
│   │   ├── index.css                   # Glassmorphism Design Tokens & CSS
│   │   └── App.css                     # Layout Stylesheet
│   └── package.json
├── tests/
│   └── test_color_classifier.py        # Standalone Color Classification Test Suite
├── yolov8s.pt                          # YOLOv8 Model Weights
├── requirements.txt                    # Python Dependencies
└── .gitignore
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9+
- Node.js 18+

### 2. Backend Setup
Activate your virtual environment and install Python dependencies:

```powershell
# Create & activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

Run the FastAPI backend server:

```powershell
$env:PYTHONPATH="."
python -m uvicorn backend.api:app --reload --port 8000
```

### 3. Frontend Setup
Open a new terminal tab and start the Vite React development server:

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` in your browser.

---

## 🧪 Testing & Evaluation

Run the standalone color classifier test suite against vehicle crops in `dataset/crops_image/`:

```powershell
$env:PYTHONPATH="."
python tests/test_color_classifier.py
```

Run the complete visual annotation pipeline on test images:

```powershell
$env:PYTHONPATH="."
python backend/src/utils/draw.py
```

---

## 📄 API Documentation

### `POST /api/analyze`
Upload a traffic image (`multipart/form-data`) to receive annotated detection results.

**Response:**
```json
{
  "success": true,
  "summary": {
    "total_cars": 4,
    "blue_cars": 1,
    "other_cars": 3,
    "people_count": 12
  },
  "image_base64": "data:image/jpeg;base64,..."
}
```

### `GET /health`
Returns system health and GPU status.

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.
