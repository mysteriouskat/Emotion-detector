# Emotion Detector

A deep learning-based application that detects human emotions from facial expressions using a webcam feed. This project utilizes facial detection and a convolutional neural network (CNN) to classify emotions in real time, displayed through a Streamlit web interface.

---

## Features

- Real-time emotion detection from webcam using `cv2` and `MTCNN`
- Preprocessing and data augmentation via TensorFlow/Keras
- Web interface built with Streamlit
- Model training and evaluation using scikit-learn
- Visualization and analytics using matplotlib, seaborn

---

## Tech Stack

This project was built using:

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, MTCNN, PIL
- **Data Processing**: Pandas, NumPy
- **Machine Learning Utilities**: Scikit-learn, joblib
- **Visualization**: Matplotlib, Seaborn
- **Others**: tqdm, os

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/mysteriouskat/Emotion-detector.git
cd Emotion-detector
```

2. Create and activate a virtual environment (optional)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. Install the required packages

```bash
pip install -r requirements.txt
```

Make sure your Python version is 3.8+.
## How to Run the App

Once dependencies are installed, you can launch the Streamlit app using:

```bash
streamlit run src/app.py
```
- You want to mention **dataset source** or **model architecture** used

I'll update accordingly.
