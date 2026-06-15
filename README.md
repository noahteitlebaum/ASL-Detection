# ASL Alphabet Detection System

A real-time American Sign Language (ASL) alphabet recognition system using computer vision and machine learning. This project uses MediaPipe for hand landmark detection and a Multi-Layer Perceptron (MLP) neural network for letter classification.

## 🎯 Features

- **Real-time ASL Detection**: Recognizes ASL alphabet letters (A-Z) from webcam feed
- **High Accuracy**: Achieves 95%+ confidence threshold for predictions
- **Custom Data Collection**: Tools to collect and process your own training data
- **Data Augmentation**: Automatically augments training data for better model performance
- **Pre-trained Model**: Includes trained model ready for immediate use

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- Windows/Linux/macOS

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/noahteitlebaum/ASL-Detection
cd ASL-Detection
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` - Image processing and webcam handling
- `mediapipe==0.10.14` - Hand landmark detection
- `scikit-learn` - Machine learning model training
- `numpy` - Numerical computations

## 📁 Project Structure

```
ASL-Detection/
├── asl_detector.py              # Real-time ASL detection (main application)
├── train_model.py               # Model training and evaluation
├── collect_data.py              # Data collection from images
├── augment_data.py              # Data augmentation utilities
├── normalize_flatten_landmarks.py  # Landmark preprocessing
├── draw.py                      # Visualization utilities
├── requirements.txt             # Project dependencies
├── data/
│   └── asl_data.json           # Training data (hand landmarks)
├── images/
│   ├── a/                      # Images for letter 'A'
│   ├── b/                      # Images for letter 'B'
│   └── ...                     # Images for other letters
├── models/
│   ├── asl_mlp_model.pkl       # Trained MLP model
│   └── label_encoder.pkl       # Label encoder for predictions
└── venv/                       # Virtual environment (created during setup)
```

## 🎮 Usage

### Real-Time ASL Detection

Run the main detection application:

```bash
python asl_detector.py
```

**Controls:**
- Position your hand in front of the webcam
- Make ASL alphabet signs
- The predicted letter appears on screen when confidence > 95%
- Press `q` to quit

### Training the Model

If you want to retrain the model with your own data:

```bash
python train_model.py
```

**What it does:**
1. Loads data from `data/asl_data.json`
2. Augments the dataset (creates 5 variations per sample)
3. Trains an MLP neural network
4. Evaluates model performance
5. Saves trained model to `models/asl_mlp_model.pkl`

**Model Architecture:**
- Input: 63 features (21 hand landmarks × 3 coordinates)
- Hidden layers: 128 → 64 → 32 neurons (ReLU activation)
- Output: 26 classes (A-Z)
- Optimizer: Adam with adaptive learning rate

### Collecting Training Data

To collect your own training data from images:

1. Organize images in folders by letter:
   ```
   images/
   ├── a/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── b/
   │   └── ...
   ```

2. Run the data collection script:
   ```bash
   python collect_data.py
   ```

3. The script will:
   - Process all images in `images/` directory
   - Extract hand landmarks using MediaPipe
   - Save data to `data/asl_data.json`

## 🧠 How It Works

### 1. Hand Landmark Detection
- Uses MediaPipe Hands to detect 21 hand landmarks
- Each landmark has (x, y, z) coordinates

### 2. Preprocessing
- Normalizes landmarks to remove position/size variations
- Flattens 21×3 landmarks into 63-feature vector

### 3. Data Augmentation
- Creates variations with rotation, scaling, and noise
- Increases dataset size 5x for better generalization

### 4. Model Training
- MLP classifier with 3 hidden layers
- 80/20 train/test split with stratified sampling
- Early stopping to prevent overfitting

### 5. Real-Time Prediction
- Captures webcam frames
- Detects hand landmarks
- Predicts letter with confidence score
- Displays result when confidence > 95%

## 🔧 Troubleshooting

### Webcam Not Working
- Ensure webcam is connected and not used by another application
- Try changing camera index in `asl_detector.py`: `cv2.VideoCapture(1)` instead of `0`

### Low Prediction Accuracy
- Ensure good lighting conditions
- Keep hand clearly visible and centered
- Retrain model with more diverse training data
- Adjust confidence threshold in `asl_detector.py`

### Import Errors
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### MediaPipe Version Issues
- The project uses `mediapipe==0.10.14` specifically
- If you encounter issues, ensure this exact version is installed

## 📊 Model Performance

The trained model achieves:
- **Test Accuracy**: ~95%+ (varies based on training data)
- **Confidence Threshold**: 95% for real-time predictions
- **Training Time**: ~2-5 minutes (depends on dataset size)

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📝 Notes

- The model works best with clear, well-lit images
- Static hand poses work better than motion
- Some letters (like J and Z) require motion and may not be detected accurately
- The system detects one hand at a time

## 🎓 Educational Purpose

This project demonstrates:
- Computer vision with OpenCV and MediaPipe
- Machine learning with scikit-learn
- Real-time video processing
- Data preprocessing and augmentation
- Neural network training and evaluation

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

---

**Happy Signing! 🤟**
