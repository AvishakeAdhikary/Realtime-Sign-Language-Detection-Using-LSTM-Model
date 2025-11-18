# Sign Language Detection with LSTM and Mediapipe

A real-time sign language detection system that uses Mediapipe Holistic for pose, hand, and face landmark detection, combined with a TensorFlow LSTM model for sequence prediction.

## 🚀 Features

- **Real-time Processing**: Webcam-based sign language detection
- **Multi-modal Input**: Utilizes pose, face, and hand landmarks
- **Easy Training**: Simple dataset collection and model training
- **Flexible Deployment**: Works with both live webcam and pre-recorded videos

## 🛠️ Tech Stack

- **Mediapipe Holistic** - For pose, hands, and face landmark detection
- **TensorFlow LSTM** - For sequence modeling and prediction
- **OpenCV** - For video capture and processing
- **uv** - For fast Python environment and dependency management

## 📦 Installation

### 1. Install uv

#### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Set Up Environment

```bash
# Create virtual environment
uv venv

# Activate environment
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

```bash
uv run main.py
# or if environment is already activated
python main.py
```

## 📊 Technical Details

### Feature Vector Composition (1662 dimensions)

| Component     | Landmarks | Values/Landmark | Total  |
|---------------|-----------|-----------------|--------|
| Pose         | 33        | 4 (x,y,z,visibility) | 132    |
| Face         | 468       | 3 (x,y,z)       | 1,404  |
| Left Hand    | 21        | 3 (x,y,z)       | 63     |
| Right Hand   | 21        | 3 (x,y,z)       | 63     |
| **TOTAL**    | **543**   | **-**           | **1,662** |

### Model Architecture

```python
LSTM(64)       # Initial temporal processing
LSTM(128)      # Main pattern recognition
LSTM(64)       # Compression layer
Dense(64)      # Feature refinement
Dense(32)      # Further refinement
Dense(#classes) # Output layer
```

### Key Parameters

- **Sequence Length**: 30 frames (~1 second at 30 FPS)
- **Sequences per Sign**: 30
- **Total Frames per Sign**: 900 (30 sequences × 30 frames)
- **Training Epochs**: 2000 (adjustable)

## 📂 Directory Structure

```
MP_Data/
    SIGN_1/
        0/             # Sequence 0
            0.npy      # Frame 0 landmarks
            ...
            29.npy     # Frame 29 landmarks
        1/             # Sequence 1
            ...
    SIGN_2/
        ...
```

## 🎯 Modes of Operation

### 1. Dataset Collection
- Captures and saves Mediapipe landmarks as `.npy` files
- Organizes data by sign and sequence

### 2. Model Training
- Loads pre-collected landmark data
- Trains LSTM model
- Saves trained model as `model.h5` and weights as `model_weights.h5`

### 3. Live Detection (Webcam)
- Real-time sign language detection
- Displays prediction probabilities
- Press 'q' to quit

### 4. Video File Detection
- Processes pre-recorded videos
- Same detection capabilities as live mode

## 💡 Tips for Best Results

### Recording
- Ensure hands are fully visible in frame
- Use good, even lighting
- Maintain consistent hand positioning
- Keep background simple and uncluttered

### Training
- Collect at least 30 sequences per sign
- Consider increasing sequence length for complex gestures
- Monitor training loss to determine optimal epoch count

### Performance
- For faster training, reduce sequence length or number of epochs
- For better accuracy, increase sequences per sign
- Use a CUDA-enabled GPU for faster training

## 📝 Notes

- The visualization uses fixed dimensions (bar height: 25px, spacing: 35px, max width: 300px)
- Model performance depends heavily on training data quality and quantity
- For production use, consider fine-tuning hyperparameters based on your specific use case
