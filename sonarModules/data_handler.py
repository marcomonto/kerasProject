import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

def setup_reproducibility() -> None:
    """Configura i seed per la riproducibilità dei risultati"""
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)

def load_and_prepare_sonar_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Carica e prepara il dataset sonar per classificazione binaria
    
    Dataset: 60 features sonar + 1 label (M=Metal, R=Rock)
    Task: Classificare segnali sonar tra cilindro metallico (M) e roccia (R)
    
    Returns:
        Tuple contenente X_train, X_test, y_train, y_test, encoder
    """
    print("Caricamento del dataset sonar...")
    
    # Carica dataset (no header, ultima colonna è il target)
    dataframe = pd.read_csv('sonar/sonar.csv', header=None)
    dataset = dataframe.values
    
    print(f"Dataset shape: {dataset.shape}")
    
    # Separa features (60 colonne) e target (ultima colonna)
    X = dataset[:, 0:60].astype(float)  # 60 features numeriche
    y = dataset[:, 60]                  # Target: 'M' o 'R'
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Classi uniche: {np.unique(y)}")
    
    # Conta le classi
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"Classe '{cls}': {count} campioni")
    
    # Encoding delle label: M=1, R=0
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)
    
    
    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded_y, 
        test_size=0.20, 
        random_state=42,
        stratify=encoded_y  # Mantiene proporzioni delle classi
    )
    
    print(f"\nTraining set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # Statistiche training set
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    print(f"Distribuzione training:")
    for cls, count in zip(encoder.classes_[train_unique], train_counts):
        print(f"  {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, encoder

def get_sonar_dataset_info() -> dict:
    """
    Restituisce informazioni sul dataset sonar
    
    Returns:
        Dizionario con informazioni del dataset
    """
    return {
        'features_count': 60,
        'classes': ['R', 'M'],
        'class_names': {'R': 'Rock (Roccia)', 'M': 'Metal (Metallo)'},
        'task_type': 'binary_classification',
        'description': 'Discriminazione segnali sonar: cilindro metallico vs roccia cilindrica',
        'feature_description': '60 valori numerici rappresentanti segnali sonar riflessi'
    }

def apply_feature_scaling(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Applica standardizzazione alle features (opzionale per migliorare performance)
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features standardizzate:")
    print(f"  Training mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    print(f"  Test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler