# AI Vision Image Classifier

A modular PyTorch-based pipeline for image classification. Includes data cleaning, optimized training with best-model saving, and a Streamlit UI.

## 📂 Project Structure

```text
vision-classifier/
├── data/               # Raw images (e.g., data/Cat, data/Dog)
├── models/             # Saved .pth model checkpoints
├── src/
│   ├── data_setup.py   # Manual 70/30 split and transforms
│   ├── model.py        # Model architecture and loading
│   ├── train.py        # Training and testing logic
│   ├── engine.py       # Main entry point for training
│   ├── predict.py      # Single image inference logic
│   └── clean_data.py   # Strict corruption/truncation cleaning
├── app.py              # Streamlit Web UI
├── .gitignore          # Git exclusion rules
└── requirements.txt    # Project dependencies
```

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vision-classifier.git
cd vision-classifier
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .env
```

**Windows:**

```bash
.env\Scripts\activate
```

**macOS/Linux:**

```bash
source .env/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install streamlit tqdm pillow
```

## 📖 Usage

### 1. Data Cleaning

Remove corrupt or truncated images:

```bash
python src/clean_data.py
```

### 2. Training

The training script automatically:

- Splits the dataset into 70% training and 30% testing
- Applies data augmentation to the training set
- Trains the model
- Evaluates the model
- Saves the best model to `models/model.pth`

Run:

```bash
python src/engine.py
```

### 3. Web Interface

Launch the Streamlit application:

```bash
streamlit run app.py
```

## 🧠 Model Configuration

| Configuration        | Details                                                |
| -------------------- | ------------------------------------------------------ |
| **Preprocessing**    | Resized to `224x224`                                   |
| **Augmentation**     | Random Horizontal Flip, Rotation (`15°`), Color Jitter |
| **Optimizer**        | SGD                                                    |
| **Scheduler**        | StepLR                                                 |
| **Data Split**       | Fixed manual `70/30` train/test split                  |
| **Model Checkpoint** | `models/model.pth`                                     |

The fixed manual split ensures consistent evaluation across runs.

## ⚠️ Git Note

The following directories are ignored by Git:

```text
data/
.env/
```

You must provide your own dataset.

## 📌 Dataset Structure

Place your images inside class-specific directories:

```text
data/
├── Cat/
│   ├── cat001.jpg
│   ├── cat002.jpg
│   └── ...
└── Dog/
    ├── dog001.jpg
    ├── dog002.jpg
    └── ...
```

Additional classes can be added using the same directory structure.

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/yourusername/vision-classifier.git
cd vision-classifier

# Create virtual environment
python -m venv .env

# Windows
.env\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install streamlit tqdm pillow

# Clean dataset
python src/clean_data.py

# Train model
python src/engine.py

# Launch web interface
streamlit run app.py
```
