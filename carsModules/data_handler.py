import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple

def setup_reproducibility() -> None:
    """Configura i seed per la riproducibilità dei risultati"""
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)

def load_and_prepare_cars_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica e prepara il dataset cars per la regressione
    
    Features:
    - Age: Età del cliente
    - Gender: Genere (0=Female, 1=Male)
    - Average miles driven per day: Miglia guidate al giorno
    - Personal debt: Debito personale
    - Monthly income: Reddito mensile
    
    Target:
    - Car price: Prezzo dell'auto (da predire)
    
    Returns:
        Tuple contenente X_train, X_test, y_train, y_test
    """
    print("Caricamento del dataset cars...")
    
    # Carica il dataset
    dataset = pd.read_csv('cars/cars.csv')
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Colonne: {list(dataset.columns)}")
    
    # Features (prime 5 colonne)
    X = dataset.iloc[:, 0:5].values
    # Target (ultima colonna - Car price)
    y = dataset.iloc[:, 5].values
    
    # Informazioni sui dati
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    print(f"Target mean: {y.mean():.2f}")
    
    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def get_feature_info() -> dict:
    """
    Restituisce informazioni sulle features del dataset
    
    Returns:
        Dizionario con informazioni sulle features
    """
    return {
        'features': [
            'Age',
            'Gender', 
            'Average miles driven per day',
            'Personal debt',
            'Monthly income'
        ],
        'target': 'Car price',
        'feature_count': 5,
        'task_type': 'regression'
    }