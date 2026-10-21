# SeeFood 🌭 vs 🚫🌭

A full-stack, transfer-learning image classification system inspired by the classic *Silicon Valley* application. SeeFood classifies food photos into **Hotdog**, **Not Hotdog**, or **Uncertain** using a fine-tuned MobileNetV2 neural network, provides model interpretability via Grad-CAM heatmaps, serves predictions through a FastAPI backend, and presents an interactive web interface built with React and Vite.

---

## Table of Contents

- [Features](#features)
- [Demo Workflow](#demo-workflow)
- [Dataset Organization and Local Inference](#dataset-organization-and-local-inference)
- [Model Architecture and Transfer Learning](#model-architecture-and-transfer-learning)
- [Data Augmentation and Training Methodology](#data-augmentation-and-training-methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Confidence Thresholds and Limitations](#confidence-thresholds-and-limitations)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [TensorFlow Lite Optimization and Benchmarking](#tensorflow-lite-optimization-and-benchmarking)
- [Technical Challenges and Solutions](#technical-challenges-and-solutions)
- [System Architecture and Directory Tree](#system-architecture-and-directory-tree)
- [Technology Stack](#technology-stack)
- [Prerequisites and Installation](#prerequisites-and-installation)
- [Model-Weight Setup](#model-weight-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Running Automated Tests](#running-automated-tests)
- [Reproducing Model Export and Benchmarks](#reproducing-model-export-and-benchmarks)
- [Known Limitations and Future Improvements](#known-limitations-and-future-improvements)

---

## Features

- **Binary & Three-Way Classification**: Distinguishes between Hotdogs and Not Hotdogs with an explicit **Uncertain** outcome for ambiguous predictions.
- **Visual Explainability (Grad-CAM)**: Generates class-activation heatmaps highlighting spatial regions that influenced the model's prediction.
- **Dual-Threshold Decision Logic**: Employs configurable bounds (`threshold_low` and `threshold_high`) loaded from JSON configuration to guard against overconfident ambiguous inferences.
- **Robust File Validation**: Inspects incoming binary streams with Pillow, verifying format integrity across JPEG, PNG, and WebP, rejecting corrupted uploads, empty files, and unsupported MIME types.
- **FastAPI REST API**: Self-documenting backend with endpoints for prediction, service health, and OpenAPI documentation (`/docs`, `/redoc`).
- **Interactive React UI**: Clean client built with React 19 and Vite 8 featuring drag-and-drop upload, image preview, prediction progress bars, live backend health monitoring, and session prediction history.
- **Edge Deployment Ready**: Standalone conversion pipeline to export quantized TensorFlow Lite flatbuffers (~74% file size reduction).

---

## Demo Workflow

1. **Check Backend Status**: When loaded, the React frontend pings `GET /health` and displays a connection status badge (`Backend Connected` or `Backend Offline`).
2. **Select an Image**: Drag and drop a food photograph or browse files from your computer (accepted: `.jpg`, `.jpeg`, `.png`, `.webp`).
3. **Trigger Inference**: Click **Classify Image**. The client packages the image as `multipart/form-data` and sends it to `POST /predict`.
4. **Inspect Prediction**:
   - For high probability ($\ge 0.65$): Classified as **Hotdog** with calculated confidence.
   - For low probability ($\le 0.35$): Classified as **Not Hotdog** with calculated confidence.
   - For mid-range probability ($0.35 < p < 0.65$): Classified as **Uncertain** with confidence set to `null` to avoid misleading values.
5. **Inspect Attention Heatmap**: Examine the side-by-side comparison of the original image alongside the Grad-CAM heatmap overlay to see where the network focused.
6. **Revisit Session History**: Review previous predictions in the current session list, switching between earlier classifications without re-querying the server.

---

## Dataset Organization and Local Inference

### Expected Dataset Directory Structure for Training
The model was trained and evaluated on an image dataset partitioned into standard train, validation, and holdout test splits, structured with dedicated subdirectories for each binary class:

```
dataset/
├── train/
│   ├── hot_dog/
│   └── not_hot_dog/
├── validation/
│   ├── hot_dog/
│   └── not_hot_dog/
└── test/
    ├── hot_dog/
    └── not_hot_dog/
```

### Acquiring and Placing the Dataset
- **Dataset Archive & Source**: The original image dataset files, annotations, and licensing metadata are not included in this repository. Any dataset archive or source information must be provided directly by the project owner.
- **Retraining Placement**: To retrain or fine-tune models from scratch, obtain the dataset archive from the project owner and extract it into a local directory matching the directory structure shown above.

### Running Local Inference Without the Training Dataset
You do **not** need the original training dataset to run inference locally. The application's inference engine and frontend interface only require:
1. The trained model binary (`models/seefood_augmented_best.keras`).
2. The runtime configuration file (`config/thresholds.json`).
3. Python dependencies in `requirements.txt` and Node dependencies in `frontend/package.json`.

Once the model file is placed in `models/`, the FastAPI backend and React frontend can process any user-uploaded JPEG, PNG, or WebP image independently.

---

## Model Architecture and Transfer Learning

### Architecture Description
The vision model leverages transfer learning built on top of **MobileNetV2** (`mobilenetv2_1.00_224`) pretrained on ImageNet:
1. **Input Layer**: Accepts RGB image tensors of shape `(None, 224, 224, 3)` with float32 values scaled to $[-1, 1]$.
2. **Backbone (Feature Extractor)**: Truncated MobileNetV2 backbone producing high-level feature maps from its terminal convolutional block (`out_relu`).
3. **Global Average Pooling**: `GlobalAveragePooling2D` layer collapsing spatial feature dimensions $(7 \times 7 \times 1280)$ into a 1D feature vector of length $1280$.
4. **Regularization Layer**: `Dropout` layer applied after global pooling to mitigate feature co-adaptation during classification head training.
5. **Classification Head**: `Dense(1, activation='sigmoid')` layer emitting a scalar sigmoid probability $p \in [0.0, 1.0]$, representing the likelihood that the input image contains a hotdog.

### Rationale for Architectural Choices
- **MobileNetV2 Backbone**: Ideal for lightweight, edge-oriented, and interactive web applications. Its inverted residual blocks and depthwise separable convolutions offer strong spatial feature representation with a low parameter count (~2.2M to 3.4M parameters depending on head configuration), enabling sub-100ms inference on standard CPUs without demanding dedicated GPU infrastructure.
- **Input Resolution ($224 \times 224 \times 3$)**: Matches the canonical ImageNet pretraining resolution of MobileNetV2, preserving pretrained filter geometry without interpolation artifacts.
- **MobileNetV2 Normalization**: Pixel normalization via `tf.keras.applications.mobilenet_v2.preprocess_input` scales raw pixel intensities from $[0, 255]$ to $[-1, 1]$, exactly aligning with the distribution expected by the pretrained backbone weights.
- **Sigmoid Output**: Emits a calibrated single-unit output mathematically paired with binary cross-entropy loss, naturally supporting two-class decision boundaries.

### Verified Hyperparameters and Training Configuration
- **Input Dimensions**: $224 \times 224 \times 3$
- **Input Normalization**: MobileNetV2 preprocessing (scaled to $[-1, 1]$)
- **Output Activation**: Sigmoid (1 unit)
- **Evaluation Test Set Size**: 691 holdout test images
- **Checkpoint Criteria**: Validation loss / accuracy tracking for saving best weights (`seefood_best.keras` and `seefood_augmented_best.keras`)

### Training Details Requiring Confirmation
The original training scripts or Colab training notebooks are not part of the active Git repository. As a result, the following training hyperparameter values cannot be verified from the codebase and require confirmation from the project owner:
- Optimization algorithm (e.g., Adam vs. SGD / RMSprop)
- Base learning rate and learning rate scheduling / decay policy
- Batch size during training and validation
- Maximum epoch limit and actual epoch at which early stopping or best checkpoint triggered
- Early stopping patience and monitored metric
- Dropout rate parameter (e.g., 0.2 vs. 0.5)
- Exact augmentation parameters (e.g., rotation ranges, zoom ranges, horizontal flip probability, brightness shifts)
- Base model weight freezing strategy (e.g., feature extraction with fully frozen backbone vs. partial fine-tuning of top convolutional blocks)

---

## Data Augmentation and Training Methodology

The project investigated whether data augmentation would improve generalization on food photographs featuring varied angles, backgrounds, and lighting.

### Baseline vs. Augmented Checkpoints
1. **Baseline Model (`models/seefood_best.keras`)**: Trained without data augmentation to establish a performance benchmark.
2. **Augmented Model (`models/seefood_augmented_best.keras`)**: Retrained with image augmentation transformations applied to the training dataset.

### Role of Each Model in the Repository
- **Active Backend Model**: The FastAPI service in `backend/inference.py` explicitly loads and serves `models/seefood_augmented_best.keras`.
- **Baseline Model Role**: Serves as the historical baseline checkpoint. Note that `scripts/export_tflite.py` and `scripts/benchmark_models.py` currently read `models/seefood_best.keras` by default.

---

## Evaluation Metrics

The following metrics represent recorded experimental results from offline model training and holdout evaluation. They are reported here as empirical experimental results rather than a guarantee of future operational performance.

### Augmented Model Test Performance
Evaluated on **691 holdout test images**:
- **Test Accuracy**: 97.11%
- **Test Loss**: 0.0949
- **Classification Report**:
  - **Not Hotdog**: Precision: 97.89% | Recall: 97.89% | F1-Score: 97.89%
  - **Hotdog**: Precision: 95.39% | Recall: 95.39% | F1-Score: 95.39%

### Baseline vs. Augmented Validation Comparison
Evaluated on the common validation dataset:
- **Baseline Model (`seefood_best.keras`)**: 95.51% validation accuracy
- **Augmented Model (`seefood_augmented_best.keras`)**: 95.94% validation accuracy
- **Observed Difference**: A modest validation increase of **+0.43 percentage points**. While data augmentation assisted in robustness across diverse angles, the gain is small and does not establish a statistically significant improvement on its own.

---

## Confidence Thresholds and Limitations

Standard binary classifiers enforce an arbitrary $0.5$ cutoff, forcing a binary decision even when the model is uncertain. To mitigate overconfident errors on ambiguous images, SeeFood applies dual decision thresholds loaded from `config/thresholds.json`:

```json
{
  "threshold_low": 0.35,
  "threshold_high": 0.65,
  "status": "provisional",
  "calibrated_on_validation_set": false,
  "note": "Provisional heuristic thresholds awaiting calibration on the validation set."
}
```

### Decision Rules
Let $p$ represent the model's output probability (`hotdog_probability`):
- **$p \ge 0.65$**: Label = `"Hotdog"`, Confidence = $p \times 100\%$
- **$p \le 0.35$**: Label = `"Not Hotdog"`, Confidence = $(1 - p) \times 100\%$
- **$0.35 < p < 0.65$**: Label = `"Uncertain"`, Confidence = `null`

### Limitations
- **Heuristic Nature**: The 0.35 and 0.65 bounds are provisional heuristics. They have not been mathematically calibrated via Platt scaling, isotonic regression, or temperature scaling against an empirical validation split.
- **Not an Out-of-Distribution (OOD) Detector**: These thresholds do not prevent errors on out-of-distribution non-food images. A non-food item whose visual features, color distributions, or textures resemble a hotdog bun or sausage can still trigger a false high-confidence prediction.
- **Null Confidence**: Confidence is explicitly returned as `null` for uncertain decisions in the API response to avoid presenting fabricated confidence scores to end users.

---

## Grad-CAM Explainability

SeeFood provides Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the spatial regions that most heavily influenced the network's prediction.

### Backend Implementation
1. In `backend/inference.py`, a dedicated multi-output model extracts convolutional feature activations from MobileNetV2's final conv layer (`out_relu`) alongside the final dense prediction.
2. `tf.GradientTape` records operations to calculate gradients of the target class score with respect to the `out_relu` convolutional feature maps.
3. Spatial average pooling is applied to the gradients to obtain neuron importance weights.
4. A weighted linear combination of the feature maps is computed and passed through a ReLU activation to retain features that positively support the decision.
5. The resulting heatmap is resized to $224 \times 224$, mapped through a Jet colormap, blended over the original user image ($\alpha = 0.45$), and returned as a Base64 JPEG data URL (`data:image/jpeg;base64,...`).

### Limitations
Grad-CAM is a visual interpretability technique that highlights salient convolutional activation regions. It provides insight into where the network's attention was directed, but it does **not** constitute mathematical proof of correct causal reasoning or biological visual understanding.

---

## TensorFlow Lite Optimization and Benchmarking

To evaluate edge deployment, the Keras models can be converted into dynamic-range quantized TensorFlow Lite (`.tflite`) flatbuffers using post-training quantization.

### Model Artifact Sizes
- **Keras Models**: ~9.6 MB (`seefood_best.keras`: 9.63 MB, `seefood_augmented_best.keras`: 9.62 MB)
- **TFLite Flatbuffers**: ~2.5 MB (`models/seefood.tflite`: 2.50 MB, `models/seefood_augmented.tflite`: 2.50 MB)
- **Size Reduction**: Approximately **74% size reduction** by quantizing model weights from Float32 to INT8 while dequantizing activations to Float32 dynamically during runtime.

### Model Alignment: Baseline vs. Augmented Workflows
The optimization scripts support separate, isolated workflows for both models:
- **Augmented Workflow** (`--model augmented`, default):
  - Keras Source: `models/seefood_augmented_best.keras` (the model served by the FastAPI backend)
  - TFLite Target: `models/seefood_augmented.tflite`
- **Baseline Workflow** (`--model baseline`):
  - Keras Source: `models/seefood_best.keras`
  - TFLite Target: `models/seefood.tflite`

Both workflows never overwrite each other's target artifacts.

### Synthetic Consistency vs. Real-Image Evaluation
1. **Synthetic Consistency Benchmarking** (`scripts/benchmark_models.py` without `--data-dir`):
   - Feeds deterministic pseudo-random Float32 arrays in $[-1, 1]$ directly to both models.
   - Measures raw numerical conversion divergence, binary decision consistency, three-way threshold agreement, and CPU latency.
   - **Important**: Synthetic testing measures conversion consistency only. It does **not** measure classification accuracy on real food photography.
2. **Real-Image Evaluation** (`scripts/benchmark_models.py --data-dir <path>`):
   - Accepts a real dataset directory containing `hot_dog/` and `not_hot_dog/` subfolders.
   - Applies the identical image loading, resizing to $224 \times 224$, and `mobilenet_v2.preprocess_input` normalization as `backend/preprocessing.py`.
   - Computes actual real-world classification accuracy, precision, recall, F1-scores, confusion matrices, and model-to-model agreement on genuine food imagery.

### Historical Colab Benchmark Measurements (Environment-Specific)
*Note: The following latency figures were measured in a specific Google Colab CPU/GPU runtime environment on the baseline model and represent historical observations, not universal hardware guarantees:*
- **Keras Mean Latency**: ~150.87 ms per inference
- **TFLite Mean Latency**: ~21.16 ms per inference (observed ~7.1x speedup in that Colab environment)
- **Decision Agreement**:
  - **98.55%** binary prediction agreement ($p \ge 0.5$) across the evaluated sample distribution.
  - **97.39%** three-way label agreement when evaluated against the dual thresholds.

---

## Technical Challenges and Solutions

### 1. Model Training Experiment: Generalization and Overfitting
- **Challenge**: Transfer learning with lightweight backbones on binary food classification risks overfitting to incidental background features (such as plates, countertops, or table textures) present in the training set.
- **Experiment & Result**: An augmented model was trained to expose the network to synthetic visual variations. The validation accuracy improved from 95.51% (baseline) to 95.94% (augmented), representing a modest gain of +0.43 percentage points. On holdout test data, the augmented model demonstrated strong performance (97.11% accuracy, 0.0949 loss).
- **Scope**: While augmentation assisted in training stability, repository records do not provide evidence that it completely eliminated overfitting or solved class imbalance. Borderline food items and non-food artifacts still required the addition of the dual-threshold "Uncertain" category.

### 2. Deployment & Optimization Challenge: Keras 3 to TFLite MLIR Conversion
- **Challenge**: In modern environments with Keras 3 and TensorFlow 2.16+, invoking `tf.lite.TFLiteConverter.from_keras_model(model)` fails during graph compilation with the following error:
  ```
  LLVM ERROR: Failed to infer result type(s).
  ```
  This failure occurs because Keras 3's internal tracing path generates `ReadVariableOp` nodes without the `value` attribute that TensorFlow Lite's MLIR frontend requires.
- **Solution**: As implemented in `scripts/export_tflite.py`, the conversion pipeline circumvents this by employing a two-step export process:
  1. The Keras 3 model is exported to an intermediate TensorFlow 2 SavedModel directory via `model.export(tmp_dir)`.
  2. The converter is initialized via `tf.lite.TFLiteConverter.from_saved_model(tmp_dir)`, with dynamic-range optimization (`converter.optimizations = [tf.lite.Optimize.DEFAULT]`). This path bypasses Keras 2 tracing and exports a valid TFLite flatbuffer cleanly for both baseline and augmented models.

---

## System Architecture and Directory Tree

```
SeeFood/
├── .gitignore                         # Excludes virtual environments, models, caches
├── requirements.txt                   # Backend and testing Python dependencies
├── README.md                          # Comprehensive project documentation
├── backend/
│   ├── main.py                        # FastAPI endpoints, CORS, file validation
│   ├── inference.py                   # Model loading, Grad-CAM generation, 3-class logic
│   └── preprocessing.py               # MobileNetV2 image preprocessing pipeline (224x224)
├── config/
│   └── thresholds.json                # Low/high decision threshold configuration
├── frontend/
│   ├── package.json                   # Frontend dependencies (React 19, Vite 8)
│   ├── package-lock.json
│   ├── vite.config.js                 # Vite bundler configuration
│   ├── index.html                     # HTML root, viewport metadata, Google Font
│   └── src/
│       ├── main.jsx                   # React entry point
│       ├── App.jsx                    # Application component (state, UI, history, upload)
│       ├── App.css                    # Component layout and theme styling
│       └── index.css                  # Global styles and resets
├── models/
│   ├── seefood.tflite                 # 2.5 MB Quantized TFLite flatbuffer (baseline)
│   ├── seefood_augmented.tflite       # 2.5 MB Quantized TFLite flatbuffer (augmented)
│   ├── seefood_augmented_best.keras   # 9.6 MB Active model loaded by backend
│   └── seefood_best.keras             # 9.6 MB Baseline model
├── scripts/
│   ├── export_tflite.py               # CLI export script (--model augmented/baseline)
│   └── benchmark_models.py            # CLI benchmark & real-image evaluation script
└── tests/
    ├── conftest.py                    # Pytest fixtures (TestClient, in-memory image generator)
    ├── test_api.py                    # FastAPI endpoint and threshold logic tests
    └── test_scripts.py                # CLI path resolution and metric calculation tests
```

---

## Technology Stack

- **Backend Framework**: FastAPI 0.115+, Uvicorn, Python 3.10+
- **Machine Learning**: TensorFlow 2.16+, Keras 3, MobileNetV2, TFLite Runtime
- **Image Processing**: Pillow (PIL), NumPy
- **Frontend Framework**: React 19.3+, Vite 8.3+, Vanilla CSS (Plus Jakarta Sans font)
- **Automated Testing**: Pytest 8.3+, HTTPX (`fastapi.testclient.TestClient`)

---

## Prerequisites and Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher (with npm)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/SeeFood.git
cd SeeFood
```

### 2. Set Up Python Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## Model-Weight Setup

Because binary model files exceed standard version control limits, they are excluded from Git via `.gitignore`. You must obtain the pre-trained weights from the project owner and place them into the `models/` directory:

```bash
# Ensure the models/ directory exists in the project root
mkdir -p models
```

Model placement requirements:
- **`models/seefood_augmented_best.keras`** *(Required to run the FastAPI backend)*
- **`models/seefood_best.keras`** *(Required only if executing `scripts/export_tflite.py` or `scripts/benchmark_models.py`)*
- **`models/seefood.tflite`** *(Optional: produced by the export script or placed manually)*

If `models/seefood_augmented_best.keras` is missing, the backend will raise a `FileNotFoundError` when `backend/inference.py` initializes.

---

## Running the Application

### 1. Start the Backend API Server
From the project root directory with the virtual environment activated:
```bash
python -m uvicorn backend.main:app --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

### 2. Start the Frontend Development Server
In a separate terminal, navigate to the `frontend/` directory:
```bash
cd frontend
npm run dev
```
- Frontend Web App: `http://localhost:5173`

The frontend communicates with the FastAPI backend at `http://127.0.0.1:8000` (configured with CORS support in `backend/main.py`).

---

## API Documentation

### Endpoints

#### 1. `GET /`
Provides a welcome message and links to API documentation.
- **Response `200 OK`**:
  ```json
  {
    "message": "Welcome to the SeeFood API! Ready to classify Hotdogs.",
    "docs_url": "/docs",
    "redoc_url": "/redoc"
  }
  ```

#### 2. `GET /health`
Returns system health, model readiness, and the active decision thresholds.
- **Response `200 OK`**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "thresholds": {
      "low": 0.35,
      "high": 0.65,
      "status": "provisional"
    }
  }
  ```

#### 3. `POST /predict`
Classifies an uploaded image, calculates probability and confidence, and generates a Grad-CAM overlay.
- **Headers**: `Content-Type: multipart/form-data`
- **Body Form Field**: `file` (Binary image file: JPEG, PNG, or WebP)

**Example cURL Request**:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_photo.jpg"
```

**Example Response (`200 OK` - Decisive Prediction)**:
```json
{
  "label": "Hotdog",
  "hotdog_probability": 0.9418293237686157,
  "confidence": 94.18293237686157,
  "gradcam_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/..."
}
```

**Example Response (`200 OK` - Uncertain Prediction)**:
```json
{
  "label": "Uncertain",
  "hotdog_probability": 0.5120481252670288,
  "confidence": null,
  "gradcam_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/..."
}
```

### Error Handling & Validation
- **`400 Bad Request`**: Uploaded file is empty or corrupted (fails Pillow verification).
  ```json
  { "detail": "Uploaded file is not a valid or readable image." }
  ```
- **`415 Unsupported Media Type`**: Image format is not in `{JPEG, PNG, WEBP}` (e.g., uploading a GIF).
  ```json
  { "detail": "Unsupported image format 'GIF'. Supported formats are: JPEG, PNG, WEBP." }
  ```
- **`422 Unprocessable Entity`**: Request omitted the required multipart `file` field.
- **`500 Internal Server Error`**: Unexpected error during image preprocessing or inference execution.

---

## Running Automated Tests

The automated test suite in `tests/` uses `pytest` and `fastapi.testclient.TestClient` to validate API routes, file handling, response schemas, threshold logic, and optimization CLI utilities without reading images from disk.

Run the test suite from the root directory:
```bash
# Recommended command within virtual environment
python -m pytest

# Or with verbose output
python -m pytest -v
```

### Test Scope
- Root (`/`) and health check (`/health`) endpoint responses.
- In-memory synthetic image testing across JPEG, PNG, and WebP formats.
- Validation that `confidence` is numeric for decisive predictions and `null` for `Uncertain` outcomes.
- HTTP error code assertions for corrupt files (400), empty files (400), unsupported formats (415), and missing fields (422).
- Unit tests for boundary conditions in `classify_probability()` and JSON fallback logic in `load_threshold_config()`.
- CLI path resolution, model configuration dictionaries, metric computations (`compute_binary_metrics`), and missing-dataset safety handling in `tests/test_scripts.py`.

---

## Reproducing Model Export and Benchmarks

### 1. Exporting to TensorFlow Lite

#### Export the Active Augmented Model (Default)
Converts `models/seefood_augmented_best.keras` into `models/seefood_augmented.tflite`:
```bash
python scripts/export_tflite.py --model augmented
```

#### Export the Baseline Model
Converts `models/seefood_best.keras` into `models/seefood.tflite`:
```bash
python scripts/export_tflite.py --model baseline
```

Both targets are saved separately in `models/` without overwriting each other.

---

### 2. Running Model Benchmarks

#### A. Synthetic Conversion-Consistency Benchmark
Compares latency, file sizes, numerical probability deviations, binary decision matching, and three-way threshold agreement across deterministic synthetic inputs:

- **Augmented Model Pair**:
  ```bash
  python scripts/benchmark_models.py --model augmented
  ```
- **Baseline Model Pair**:
  ```bash
  python scripts/benchmark_models.py --model baseline
  ```

*Note: Synthetic inputs test conversion fidelity between Keras and TFLite; they do not calculate food classification accuracy.*

#### B. Real Validation/Test Image Evaluation
To measure genuine real-world accuracy, precision, recall, F1-scores, confusion matrices, and latency on real photographs, supply a dataset directory structured with `hot_dog/` and `not_hot_dog/` subfolders:

- **Evaluate Augmented Model Pair**:
  ```bash
  python scripts/benchmark_models.py --model augmented --data-dir path/to/dataset/validation
  ```
- **Evaluate Baseline Model Pair**:
  ```bash
  python scripts/benchmark_models.py --model baseline --data-dir path/to/dataset/validation
  ```

---

## Known Limitations and Future Improvements

- **Empirical Threshold Calibration**: The 0.35 and 0.65 decision thresholds are heuristic. Future work will calibrate thresholds against validation set ROC/PR curves or apply Platt scaling.
- **Out-of-Distribution Handling**: Train an explicit background rejection class or hierarchical classifier to detect non-food imagery.
- **Automated Model Weight Distribution**: Provide an automated script with checksum validation to download model binaries from release storage (such as GitHub Releases or Hugging Face Hub).
