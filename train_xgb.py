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
# Module 4: Extreme Gradient Boosting (XGBoost) Classification
# =====================================================================
def train_xgboost(X_train_encoded, y_train):
    """
    Configures and trains the multi-class Extreme Gradient Boosting classifier
    utilizing the 16-dimensional latent space representations.
    """
    print(" Initializing XGBoost Classifier Architecture...")
    
    # Define rigorous hyperparameters for multi-class tree boosting
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',  # Outputs probability distribution across all classes
        num_class=11,                # 11 distinct operational classes
        learning_rate=0.1,           # Shrinkage factor to prevent overfitting
        n_estimators=200,            # Maximum number of sequential boosting rounds
        max_depth=6,                 # Strict limit on decision tree depth
        subsample=0.8,               # Randomly sample 80% of rows per tree
        colsample_bytree=0.8,        # Randomly sample 80% of latent features per tree
        n_jobs=-1,                   # Utilize all available CPU cores for parallel processing
        random_state=42
    )
    
    print(" Executing XGBoost Training on 16-Dimensional Latent Feature Space...")
    xgb_model.fit(X_train_encoded, y_train)
    print(" XGBoost Model Convergence Complete.")
    
    return xgb_model