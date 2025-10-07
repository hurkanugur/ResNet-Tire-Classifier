# 🛞 ResNet Tire Classifier

## 📖 Overview
This project predicts **tire classes** (e.g., damaged vs. normal) using **Transfer Learning** with a pretrained **ResNet50** architecture from **PyTorch**.  
It demonstrates a modern deep learning pipeline including:

- 🧠 **Pretrained ResNet50** backbone from ImageNet for feature extraction  
- 🧩 **Custom Fully Connected (FC) Classifier Head** with **Batch Normalization**, **ReLU**, and **Dropout**  
- ⚖️ **Cross-Entropy Loss** for binary classification  
- 🚀 **Adam optimizer** for efficient training  
- 🔒 **Frozen ResNet backbone** to leverage pre-learned visual features  
- 🧰 **Modular design** — easily switch between saving only the FC head or the full model  
- 📊 **Configurable hyperparameters** via `config.py`

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **torchvision** – pretrained ResNet50 weights
- **pandas**, **numpy** – data handling
- **matplotlib** – loss visualization  
- **pickle** – saving/loading normalization params and trained model

---

## ⚙️ Requirements

- Python **3.13+**
- Recommended editor: **VS Code**

---

## 📦 Installation

- Clone the repository
```bash
git clone https://github.com/hurkanugur/ResNet-Tire-Classifier.git
```

- Navigate to the `ResNet-Tire-Classifier` directory
```bash
cd ResNet-Tire-Classifier
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Navigate to the `ResNet-Tire-Classifier/src` directory
```bash
cd src
```

---

## 🔧 Setup Python Environment in VS Code

1. `View → Command Palette → Python: Create Environment`  
2. Choose **Venv** and your **Python version**  
3. Select **requirements.txt** to install dependencies  
4. Click **OK**

---

## 📂 Project Structure

```bash
data/
└── defective                         # defective tire images
└── good                              # good tire images

model/
└── tire_classifier.pth               # Trained model (Custom FC layer weights)

src/
├── config.py                         # Paths, hyperparameters, split ratios
├── dataset.py                        # Data loading & preprocessing
├── device_manager.py                 # Selects and manages compute device
├── main_train.py                     # Training & model saving
├── main_inference.py                 # Inference pipeline
├── model.py                          # Neural network definition
├── visualize.py                      # Training/validation plots

requirements.txt                      # Python dependencies
```

---

## 📂 Model Architecture

```bash
ResNet50 (Pretrained on ImageNet)
  ↓
[Feature Extractor]                   # All convolutional layers (frozen)
  ↓
Custom Classifier Head:
  → Linear(in_features, 1024)
  → BatchNorm1d(1024)
  → ReLU
  → Dropout(0.5)
  → Linear(1024, 2)
  → Softmax(Output)
```

---

## 📂 Train the Model
```bash
python main_train.py
```
or
```bash
python3 main_train.py
```

---

## 📂 Run Predictions on Real Data
```bash
python main_inference.py
```
or
```bash
python3 main_inference.py
```
