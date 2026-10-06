# driver-drowsiness-detection-ML-project
driver drowsiness detection using python  machine learning 
# Driver Drowsiness Detection

A real-time driver drowsiness detection system that uses computer vision and a Convolutional Neural Network (CNN) to monitor the driver's eye state.

The system captures frames from a camera, detects the eye region, preprocesses the eye image, and sends it to a trained CNN model for classification. Consecutive predictions are then used to make the final drowsiness decision and trigger an alert.

The trained model was also integrated into a mobile application so that the system can operate as an edge-based solution.

## Features

- Real-time camera-based detection
- Eye-region detection and cropping
- 64 × 64 image preprocessing
- Grayscale image input
- CNN-based eye-state classification
- Temporal decision logic
- Audio alert
- Mobile edge deployment

## Dataset

The model was trained using a combination of:

- Driver Drowsiness Dataset (DDD)
- MRL Eye Dataset

The DDD images were processed to focus on the eye region, and the datasets were combined to provide more variation in eye appearance and conditions.

## Model

The project uses a custom Convolutional Neural Network (CNN).

The CNN was built and trained using TensorFlow/Keras.

Final model input:

- Image size: 64 × 64
- Image type: Grayscale
- Classes: Drowsy / Non-Drowsy
- Batch size: 64

The final eye-focused model achieved approximately 98.84% accuracy on the held-out test set.

## Project Pipeline

```text
Camera
   ↓
Face / Eye Detection
   ↓
Eye Region Crop
   ↓
Resize to 64 × 64
   ↓
Grayscale Conversion
   ↓
CNN Model
   ↓
Eye-State Prediction
   ↓
Temporal Decision
   ↓
Alert
