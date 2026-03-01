# ASL Alphabet Recognition using Deep Learning

This project is a real-time American Sign Language (ASL) alphabet recognition system. It uses a Convolutional Neural Network (CNN) trained on Kaggle datasets to predict hand gestures via a webcam.



[Image of ASL alphabet hand signs]
 

## 🚀 Features
- **Real-time Detection**: Uses OpenCV to capture video and predict gestures instantly.
- **Deep Learning Model**: Built with TensorFlow/Keras using a CNN architecture.
- **Pre-processed Images**: Hand regions are automatically cropped, grayscaled, and resized to 64x64 for optimal prediction.

## 🛠️ Technologies Used
- **Python** 3.x
- **TensorFlow / Keras** (Deep Learning Framework)
- **OpenCV** (Computer Vision)
- **Scikit-Learn** (Label Encoding)
- **NumPy** (Numerical Processing)
- **Git LFS** (For hosting the large .h5 model file)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NorikHovhannisyan/ASL-Alphabet-Recognition.git](https://github.com/NorikHovhannisyan/ASL-Alphabet-Recognition.git)
   cd ASL-Alphabet-Recognition

## ⚙️ Installation & Setup

Follow these steps to get the project running on your local machine:

### 1. Create a Virtual Environment
*(Optional but highly recommended to keep your global Python environment clean)*

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
 ### 2. Install Dependencies
Install the required libraries using pip:

```Bash
pip install tensorflow opencv-python scikit-learn numpy
```

### 3. Pull the Model
If you are using Git LFS, make sure to download the actual model file:

```Bash
git lfs pull
```

## 🖥️ Usage
Run the main script to start the webcam interface:

```Bash
python Asl.py
``` 
### 💡 Instructions:
- Positioning: Once the window opens, place your hand inside the blue square (ROI).
- Prediction: The predicted letter will appear at the top of the frame in real-time.
- Exit: Press the 'q' key on your keyboard to close the application.

## 📊 Model Performance
The model was trained on the ASL Alphabet dataset with the following results:

- Training Accuracy: ~92%
- Validation Accuracy: ~99%
- Input Shape: (64, 64, 1) — Grayscale images resized to 64x64 pixels.
