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

from evaluate import evaluate_model
from load_data import load_and_aggregate_data
from preprocess import preprocess_data
from train_autoencoder import build_and_train_autoencoder
from train_xgb import train_xgboost 

# Enforce strict random seeds to guarantee scientific reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# =====================================================================
# Module 6: Primary Execution Block
# =====================================================================
if __name__ == "__main__":
    try:
        # Step 1: Automated Ingestion of CSV files from the 'dataset/' directory
        raw_df = load_and_aggregate_data(data_dir='dataset/')
        
        # Step 2: Preprocess, Normalize, and execute SMOTE balancing
        X_train_bal, X_test_scl, y_train_bal, y_test, le = preprocess_data(raw_df)
        
        # Step 3: Train the Deep Autoencoder and retain the encoder function
        encoder_model = build_and_train_autoencoder(X_train_bal, input_dim=115)
        
        # Transform the massive 115-D training and testing matrices into condensed 16-D latent vectors
        print(" Transforming 115-dimensional statistical data into 16-dimensional latent representations...")
        X_train_latent = encoder_model.predict(X_train_bal)
        X_test_latent = encoder_model.predict(X_test_scl)
        
        # Step 4: Train the XGBoost classifier exclusively on the 16-D representations
        xgb_classifier = train_xgboost(X_train_latent, y_train_bal)
        
        # Step 5: Execute final empirical evaluation
        evaluate_model(xgb_classifier, X_test_latent, y_test, le)
        
        print(" Pipeline Execution Terminated Successfully.")
        
    except Exception as e:
        print(f"\n Pipeline execution critically failed: {str(e)}")