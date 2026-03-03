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
# Module 1: Automated Data Ingestion and Aggregation
# =====================================================================
def load_and_aggregate_data(data_dir='dataset/'):
    """
    Iterates through the target directory, identifies all CSV files representing 
    different botnet traffic classes, extracts the class label directly from the 
    filename structure, and concatenates the disparate files into a unified DataFrame.
    """
    print(" Initiating Automated Data Ingestion Protocol...")
    
    # Locate all CSV files within the designated dataset directory
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"CRITICAL ERROR: No CSV files located in '{data_dir}'. "
                                f"Please populate the directory with N-BaIoT data files.")
    
    df_list = []
    for file in all_files:
        filename = os.path.basename(file).lower()
        
        # Taxonomic Classification based on standard N-BaIoT filename conventions
        if 'benign' in filename:
            label = 'benign'
        elif 'gafgyt.combo' in filename:
            label = 'gafgyt_combo'
        elif 'gafgyt.junk' in filename:
            label = 'gafgyt_junk'
        elif 'gafgyt.scan' in filename:
            label = 'gafgyt_scan'
        elif 'gafgyt.tcp' in filename:
            label = 'gafgyt_tcp'
        elif 'gafgyt.udp' in filename:
            label = 'gafgyt_udp'
        elif 'mirai.ack' in filename:
            label = 'mirai_ack'
        elif 'mirai.scan' in filename:
            label = 'mirai_scan'
        elif 'mirai.syn' in filename:
            label = 'mirai_syn'
        elif 'mirai.udp' in filename:
            label = 'mirai_udp'
        elif 'mirai.udpplain' in filename:
            label = 'mirai_udpplain'
        else:
            print(f" Unrecognized filename topology: {filename}. Skipping file.")
            continue 
            
        # Read the individual CSV and append the designated label column
        temp_df = pd.read_csv(file)
        temp_df = label
        df_list.append(temp_df)
        
    # Concatenate all device and attack traffic into a single continuous matrix
    master_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f" Data Ingestion Complete. Unified Matrix Shape: {master_df.shape}")
    return master_df
