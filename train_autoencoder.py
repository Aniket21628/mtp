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
# Module 3: Deep Autoencoder Construction & Latent Feature Extraction
# =====================================================================
def build_and_train_autoencoder(X_train, input_dim=115):
    """
    Constructs the non-linear 3-layer encoder and 2-layer decoder architecture.
    Trains the network in a self-supervised manner to minimize MSE reconstruction loss.
    """
    print(" Constructing Deep Autoencoder Neural Network Topology...")
    
    # Input Layer corresponding to the 115 statistical features
    input_layer = Input(shape=(input_dim,))
    
    # Compressive Encoder Sub-network
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    # The absolute 16-dimensional bottleneck vector
    latent_space = Dense(16, activation='relu', name='latent_bottleneck')(encoded)
    
    # Expansive Decoder Sub-network
    decoded = Dense(32, activation='relu')(latent_space)
    decoded = Dense(64, activation='relu')(decoded)
    # Linear activation on output to map to the unbounded scaled features
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    # Compile the Autoencoder defining the Adam optimizer and MSE loss function
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Isolate and retain the Encoder portion for downstream feature extraction
    encoder = Model(inputs=input_layer, outputs=latent_space)
    
    # Configure Early Stopping to absolutely prevent model overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    print(" Initiating Autoencoder Training (Unsupervised Representation Learning)...")
    # Training utilizes X_train as BOTH the input and the target objective
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_split=0.2, # Allocate 20% of the training matrix for validation
        callbacks=[early_stopping],
        verbose=1
    )
    
    print(" Autoencoder Training Complete. Decoder sub-network discarded.")
    return encoder
