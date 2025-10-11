# Real-Time Facial Emotion Recognition with ResNet50

<div align="center">

High-performance real-time facial emotion recognition system using ResNet50 architecture, trained on FER2013 dataset.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Demo

https://github.com/user-attachments/assets/your-video-id-here

> Replace with your recorded demo video

---

## Overview

Real-time emotion detection system that identifies 7 distinct facial emotions through webcam feeds. Built on custom ResNet50 architecture with **65.59% validation accuracy** on [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

**Key Features:**
- **Real-time Processing:** 30+ FPS on GPU with optimized inference pipeline
- **Seven Emotions:** Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger
- **High Accuracy:** 65.59% validation accuracy, ~80% training accuracy
- **Production Ready:** FP16 precision, temporal smoothing, threaded capture

---

## Architecture

**Model:** Custom ResNet50 with Bottleneck blocks [3, 4, 6, 3]

```
Input (224×224×3) → Conv2dSame(7×7) → BatchNorm → ReLU → MaxPool
→ ResBlock Layer1 (64)  → ResBlock Layer2 (128) 
→ ResBlock Layer3 (256) → ResBlock Layer4 (512)
→ AdaptiveAvgPool → FC(2048→512) → FC(512→7) → Softmax
```

**Training Setup:**
- **Optimizer:** Adam (lr=0.001, weight_decay=0.0001)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Batch Size:** 64 | **Epochs:** 50
- **Augmentation:** Horizontal flip, rotation (±15°), brightness/contrast (0.8-1.2x)

---

## Performance

![Training Curves](models/training_curves.png)

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **65.59%** |
| Training Accuracy | ~80.00% |
| Inference Speed (GPU) | 30+ FPS |
| Inference Speed (CPU) | 10-15 FPS |
| Model Parameters | ~23.5M |
| Model Size | ~90 MB |

---

## Dataset

**[FER2013](https://www.kaggle.com/datasets/msambare/fer2013)** - Facial Expression Recognition benchmark dataset
- **Training:** ~28,709 images
- **Testing:** ~3,589 images
- **Classes:** 7 emotions (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger)
- **Resolution:** 48×48 grayscale (upscaled to 224×224 RGB)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam

### Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/FER2013-ResNet50-Emotion-Recognition.git
   cd FER2013-ResNet50-Emotion-Recognition
   ```

2. **Install uv** (fast package manager)
   ```bash
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create environment and install dependencies**
   ```bash
   uv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

4. **Add pre-trained model**
   
   Place the trained model file in the `models/` directory:
   ```
   models/FER_static_ResNet50_AffectNet.pt
   ```
   
   > **Note:** Model file size ~90MB. Available on request or train your own using the notebook.

---

## Usage

### Real-Time Detection

Run the emotion recognition system:

```bash
python realtime_facial_analysis.py
```

**Controls:** Press `q` to quit

**Output:**
- Real-time video feed with emotion labels and confidence scores
- FPS counter
- Face bounding boxes

### Training Custom Model

Open training notebook in Jupyter:

```bash
jupyter notebook model_trainign_resnet.ipynb
```

The notebook includes complete pipeline: data loading, model training, evaluation, and export.

---

## Project Structure

```
FER2013-ResNet50-Emotion-Recognition/
│
├── models/
│   ├── FER_static_ResNet50_AffectNet.pt
│   └── training_curves.png
├── realtime_facial_analysis.py
├── model_trainign_resnet.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technical Details

**Image Preprocessing:**
- Resize to 224×224 using INTER_NEAREST interpolation
- BGR to RGB color space conversion
- Mean normalization: [R: 91.4953, G: 103.8827, B: 131.0912]

**Performance Optimizations:**
- **GPU:** FP16 half-precision + CuDNN auto-tuning
- **Detection:** 50% downscaled resolution for face detection
- **Capture:** Threaded webcam stream (non-blocking)
- **Inference:** Temporal smoothing over 10-frame window
- **Result:** 30+ FPS on NVIDIA GPU, ~10-15 FPS on CPU

---

## Citation

```bibtex
@software{fer2013_resnet50,
  title={Real-Time Facial Emotion Recognition with ResNet50},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/FER2013-ResNet50-Emotion-Recognition}
}
```

**FER2013 Dataset:**
```bibtex
@inproceedings{goodfellow2013challenges,
  title={Challenges in representation learning},
  author={Goodfellow, Ian J and others},
  booktitle={ICONIP},
  year={2013}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **FER2013 Dataset** - ICML 2013 Challenges in Representation Learning
- **ResNet Architecture** - He et al., "Deep Residual Learning for Image Recognition"
- **MediaPipe** - Google's ML solutions for face detection
- **PyTorch** - Facebook AI Research

---

<div align="center">

**Star this repository if you find it helpful!**

</div>
