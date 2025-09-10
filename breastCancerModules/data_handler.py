from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from typing import Tuple

def setup_reproducibility() -> None:
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(60)

def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print(f"Dimensioni dataset: {X_train.shape[0]} training, {X_test.shape[0]} test")
    
    return X_train, X_test, y_train, y_test