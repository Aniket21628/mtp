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
# Module 2: Preprocessing, Z-Score Normalization, & SMOTE Balancing
# =====================================================================
def preprocess_data(df):
    """
    Executes a rigorous pipeline comprising redundancy elimination, 
    categorical label encoding, test-train segregation, statistical 
    normalization, and synthetic minority over-sampling.
    """
    print(" Commencing Advanced Preprocessing Pipeline...")
    
    # 2.1 Deduplication: Elimination of exact redundant statistical rows
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f" Deduplication purged {initial_rows - len(df)} redundant rows. "
          f"Optimized Matrix Shape: {df.shape}")
    
    # 2.2 Segregation of continuous features and discrete categorical targets
    feature_df = df.drop(['Target_Label', 'Source_File', 'Split'], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X = feature_df.values
    y_text = df['Target_Label'].astype(str).values
    
    # 2.3 Label Encoding: Transformation of string labels to integer classes (0-10)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)
    
    # 2.4 Data Segregation: 80% Training Matrix, 20% Evaluation Matrix
    # Stratification mathematically guarantees proportional class representation in both splits
    if 'Split' not in df.columns:
        raise KeyError("Split column missing. Re-run data loading with updated load_data.py.")
    train_mask = df['Split'].astype(str).eq('train').values
    test_mask = df['Split'].astype(str).eq('test').values

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y_encoded[train_mask]
    y_test = y_encoded[test_mask]
    
    # 2.5 Z-score Normalization (StandardScaler)
    # The scaler is strictly fitted ONLY on training data to prevent temporal data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2.6 Geometric Class Balancing via SMOTE
    # SMOTE is applied EXCLUSIVELY to the training matrix to preserve realistic evaluation conditions
    print(" Executing SMOTE Synthetic Over-sampling on training matrix...")
    class_counts = pd.Series(y_train).value_counts()
    majority_count = int(class_counts.max())
    target_count = int(majority_count * 0.5)

    sampling_strategy = {
        int(cls): int(target_count)
        for cls, count in class_counts.items()
        if int(count) < int(target_count)
    }

    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f" SMOTE Protocol Complete. Balanced Training Matrix Shape: {X_train_balanced.shape}")
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, label_encoder