import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_cars_regression_model(input_dim: int = 5) -> Sequential:
    """
    Crea il modello di regressione per predire il prezzo delle auto
    
    Architettura:
    - Input layer: input_dim=5 features, activation='relu'  
    - Hidden layer: activation='relu'
    - Output layer: 1 nodo per regressione
    
    Configurazione secondo specifiche:
    - loss='mse'
    - optimizer='rmsprop' 
    - metrics=['mse']
    
    Args:
        input_dim: Numero di features di input (default=5)
        
    Returns:
        Modello Keras compilato per regressione
    """
    print("Creazione modello di regressione per cars...")
    
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),  
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        loss='mse',                    # Mean Squared Error
        optimizer='rmsprop',           # RMSprop optimizer
        metrics=['mse']                # Metriche da monitorare
    )
    
    return model

def create_alternative_cars_model(input_dim: int = 5, hidden_nodes: int = 50) -> Sequential:
    """
    Modello alternativo con diversa architettura per confronto
    
    Args:
        input_dim: Numero di features di input
        hidden_nodes: Numero di nodi nel layer nascosto
        
    Returns:
        Modello Keras alternativo
    """
    print(f"Creazione modello alternativo con {hidden_nodes} nodi nascosti...")
    
    model = Sequential([
        Dense(hidden_nodes, input_dim=input_dim, activation='relu'),
        Dense(hidden_nodes//2, activation='relu'), 
        Dense(1)
    ])
    
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['mse']
    )
    
    return model