AI Vision Image Classifier

A modular PyTorch-based pipeline for image classification. Includes data cleaning, optimized training with best-model saving, and a Streamlit UI.

📂 Project Structure
vision-classifier/
├── data/ # Raw images (e.g., data/Cat, data/Dog)
├── models/ # Saved .pth model checkpoints
├── src/
│ ├── data_setup.py # Manual 70/30 split and transforms
│ ├── model.py # Model architecture and loading
│ ├── train.py # Training and testing logic
│ ├── engine.py # Main entry point for training
│ ├── predict.py # Single image inference logic
│ └── clean_data.py # Strict corruption/truncation cleaning
├── app.py # Streamlit Web UI
├── .gitignore # Git exclusion rules
└── requirements.txt # Project dependencies

🛠️ Installation

1. Clone the Repository
   git clone https://github.com/yourusername/vision-classifier.git
   cd vision-classifier

2. Set Up a Virtual Environment
   python -m venv .env

Windows:

.env\Scripts\activate

macOS/Linux:

source .env/bin/activate

3. Install Dependencies

For PyTorch with CUDA 11.8:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Install the remaining dependencies:

pip install streamlit tqdm pillow

📖 Usage

1. Data Cleaning

Remove corrupt or truncated images that could otherwise crash the training loop:

python src/clean_data.py

2. Training

Run the engine script. It automatically:

Splits the dataset into 70% training / 30% testing
Applies data augmentation to the training set
Trains the image classification model
Evaluates the model on the test set
Saves the best-performing model to models/model.pth

Run:

python src/engine.py

3. Web Interface

Launch the Streamlit application to classify images through your browser:

streamlit run app.py

🧠 Model Configuration
Configuration Details
Preprocessing Images resized to 224x224
Augmentation Random Horizontal Flip, Rotation (15°), Color Jitter
Optimizer SGD
Scheduler StepLR
Data Split Fixed manual 70/30 train/test split
Model Checkpoint models/model.pth

The fixed manual split ensures consistent evaluation across runs.

⚠️ Git Note

The following directories are ignored by Git:

data/
models/
.env/

You must provide your own dataset and manage trained model weights locally.

📌 Dataset Structure

Place your images inside class-specific directories:

data/
├── Cat/
│ ├── cat001.jpg
│ ├── cat002.jpg
│ └── ...
└── Dog/
├── dog001.jpg
├── dog002.jpg
└── ...

Additional classes can be added using the same directory structure.

🚀 Quick Start

# Clone

git clone https://github.com/yourusername/vision-classifier.git
cd vision-classifier

# Create and activate virtual environment

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
