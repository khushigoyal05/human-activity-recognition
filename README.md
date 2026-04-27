# 🏃 Kinetics AI: Human Activity Recognition (HAR)

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)

**Kinetics AI** is an advanced Deep Learning full-stack application designed to classify human physical activities using smartphone sensor data. 

This project explores, trains, and evaluates 8 different sequential deep learning architectures (including LSTMs, TCNs, and modern Transformers) on the **UCI HAR Dataset** to predict whether a user is *Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, or Laying*.

---

## 📊 The Dataset & Preprocessing Trick
The [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) consists of 561 pre-engineered statistical features extracted from Accelerometer and Gyroscope sensors at 50Hz.

### The `(11, 51)` Temporal Reshape
Because deep sequence models (like LSTMs) require 3-dimensional temporal input `(Batch, Timesteps, Features)` but the dataset provides a flat 1D array of 561 features, a mathematical reshaping technique is applied:
**`561 Features = 11 Timesteps × 51 Features-per-step`**
This forces the networks to sequentially process chunks of related sensor metrics (Acceleration → Jerk → Gyroscope → Frequency Domain) over time, perfectly mimicking chronological sensor readings.

---

## 🧠 Neural Network Architectures
The pipeline trains and evaluates the following 8 models:

1. **GRU (Winner 🏆)**: The most efficient balance of memory and parameter count. Prevented overfitting on the small dataset and achieved the highest accuracy.
2. **1D-CNN**: Extracts spatial correlations instantly using 3-step sliding windows. Extremely fast and lightweight.
3. **LSTM**: Classic Pyramidal sequence memory (128 → 64 units).
4. **BiLSTM**: Reads the sequence forward and backward for richer context.
5. **GLU (Gated Linear Units)**: Uses learnable gates to suppress irrelevant temporal noise.
6. **TCN (Temporal Convolutional Network)**: Uses Causal Dilated Convolutions to process the entire sequence in parallel without a memory loop.
7. **Transformer**: Implements Multi-Head Attention to find relationships between distant timesteps.
8. **iTransformer (ICLR 2024)**: Inverts the attention mechanism to run over *features* instead of timesteps, capturing complex cross-sensor dependencies.

### Performance Leaderboard

| Rank | Model | Accuracy | F1-Weighted | Train Time |
|---|---|---|---|---|
| 🥇 1 | **GRU** | **92.60%** | **92.59%** | 44.9s |
| 🥈 2 | GLU | 92.57% | 92.46% | 25.5s |
| 🥉 3 | TCN | 91.55% | 91.53% | 61.0s |
| 4 | LSTM | 91.04% | 91.00% | 45.1s |
| 5 | 1D-CNN | 90.94% | 90.80% | 19.0s |
| 6 | Transformer | 89.92% | 89.87% | 44.3s |
| 7 | BiLSTM | 89.14% | 89.04% | 61.1s |
| 8 | iTransformer | 86.70% | 86.48% | 80.5s |

---

## 💻 The Streamlit Dashboard
The project includes a production-ready **Streamlit Web Dashboard** featuring:
- **Lazy Loading**: Models are dynamically loaded into RAM only when selected to optimize memory.
- **Demo Test Fetcher**: Randomly pulls test cases from the unseen `X_test.txt` data.
- **Live Prediction**: Visualizes the Softmax probability distribution and feature magnitudes in real-time.
- **Analytics Portal**: Dynamic Plotly charts comparing model Accuracies, F1-Scores, and Confusion Matrices.

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/khushigoyal05/human-activity-recognition.git
cd human-activity-recognition
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed.
```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib
```

### 3. Run the Dashboard
To start the interactive frontend, simply run:
```bash
python start.py
```
*Your browser will automatically open to `http://localhost:8501`.*

### 4. Retrain the Models (Optional)
If you wish to retrain all 8 models from scratch and generate new visualization plots, run the pipeline script:
```bash
python har_pipeline.py
```
*(Warning: This will overwrite the saved models in the `/results` directory).*
