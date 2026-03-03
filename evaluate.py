"""
Comprehensive Implementation of Hybrid Deep Learning and Boosting for IoT Intrusion Detection
Architecture: Deep Autoencoder (Latent Feature Extraction) + XGBoost (Multi-Class Classification)
Dataset: N-BaIoT Benchmark (115 temporal statistical features, 11 operational classes)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Modules for Preprocessing and Empirical Evaluation
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Imbalanced-Learn Library for Geometric SMOTE Balancing
from imblearn.over_sampling import SMOTE

# TensorFlow / Keras Framework for the Deep Autoencoder Construction
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Extreme Gradient Boosting Classifier
import xgboost as xgb

# Enforce strict random seeds to guarantee scientific reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================================
# Module 5: Empirical Evaluation and Visualization
# =====================================================================
def evaluate_model(model, X_test_encoded, y_test, label_encoder):
    """
    Generates macroscopic evaluation metrics and renders the definitive Confusion Matrix.
    """
    print(" Evaluating Hybrid Architecture on Isolated Test Matrix...")
    
    # Generate discrete class predictions
    y_pred = model.predict(X_test_encoded)
    
    # Calculate global statistical metrics
    acc = accuracy_score(y_test, y_pred)
    # The 'macro' average treats all classes equally, critical for evaluated imbalanced capability
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\n" + "=" * 60)
    print("HYBRID AUTOENCODER-XGBOOST PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Overall Accuracy       : {acc * 100:.2f}%")
    print(f"Macro Precision        : {prec * 100:.2f}%")
    print(f"Macro Recall           : {rec * 100:.2f}%")
    print(f"Macro F1-Score         : {f1 * 100:.2f}%")
    print("=" * 60 + "\n")
    
    # Render and export the Confusion Matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.classes_
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Definitive Confusion Matrix - Autoencoder + XGBoost Hybrid Model', fontsize=16)
    plt.ylabel('Authentic True Class', fontsize=12)
    plt.xlabel('Algorithm Predicted Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_results.png', dpi=300)
    print(" Confusion matrix visualization successfully rendered and saved as 'confusion_matrix_results.png'.")