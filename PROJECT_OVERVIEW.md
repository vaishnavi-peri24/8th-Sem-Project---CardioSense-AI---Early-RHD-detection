# Project Overview: CardioSense AI – Low-Cost AI-Enabled Murmur Analysis Tool

This project is an end-to-end pipeline for the early detection of Rheumatic Heart Disease (RHD) using heart sound (PCG) recordings. It combines advanced signal processing, deep learning, and explainable AI to classify heart sounds as "Normal" or "Murmur" and provides interpretable results for clinical and research use.

## Key Components

1. **Dataset Collection & Organization**
   - Merges and cleans heart sound data from PhysioNet 2016 and 2022 datasets.
   - Converts all labels to a unified Normal/Murmur format.
   - Removes duplicates and validation data for robust training.

2. **Audio Preprocessing**
   - Standardizes audio by resampling, bandpass filtering (20–400 Hz), and normalization.
   - Removes corrupt or very short files.
   - Outputs cleaned audio files and metadata.

3. **Segmentation & Feature Extraction**
   - Splits audio into 3-second segments.
   - Converts each segment into a 128-band Mel spectrogram (image-like features).
   - Ensures fixed-size input for the model and saves features as .npy files.

4. **Model Training**
   - Patient-wise data split to prevent information leakage.
   - CNN model trained with class weighting to address class imbalance.
   - Early stopping and best model checkpointing for optimal performance.

5. **Evaluation & Explainability**
   - Evaluates model using accuracy, recall, AUC, confusion matrix, and ROC curve.
   - Provides S1/S2 segmentation (heart sound phases) and GradCAM visualizations for model explainability.

6. **Web Application**
   - Streamlit-based app for uploading PCG audio, running predictions, and visualizing explanations.

## Project Structure

- `app.py`: Streamlit web app for murmur detection and explainability.
- `implementation.ipynb`: Full pipeline notebook (data, training, analysis).
- `model/`: Contains the trained CNN model.
- `processed_audio/`, `spectrograms/`: Cleaned audio and extracted features.
- `2016/`, `2022/`: Raw PhysioNet datasets.
- `RHD_detection/`: Python virtual environment.

## Technologies Used

- Python, TensorFlow/Keras, NumPy, Pandas, Librosa, Matplotlib, Streamlit

## Purpose

This tool is designed for research and educational purposes, enabling low-cost, interpretable AI-based murmur detection from heart sound recordings. It is not intended for clinical diagnosis without further validation.
