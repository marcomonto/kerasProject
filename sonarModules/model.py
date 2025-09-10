import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_sonar_classification_model(input_dim: int = 60) -> Sequential:
    """
    Crea il modello di classificazione binaria per sonar secondo specifiche
    
    Specifiche richieste:
    - loss='binary_crossentropy'
    - optimizer='adam' 
    - metrics=['accuracy']
    - epochs=100, batch_size=5
    
    Architettura:
    - Input Dense layer: 60 nodi (input_dim=60, activation='relu')
    - Output Dense layer: 1 nodo (activation='sigmoid')
    
    Args:
        input_dim: Numero di features di input (default=60)
        
    Returns:
        Modello Keras compilato per classificazione binaria
    """
    print("Creazione modello classificazione binaria sonar...")
    print("Architettura: Input(60) -> Dense(60, relu) -> Dense(1, sigmoid)")
    
    model = Sequential([
        # Input layer esplicito
        Input(shape=(input_dim,)),
        
        # Input Dense layer con 60 nodi e ReLU
        Dense(60, activation='relu'),
        
        # Output Dense layer con 1 nodo e sigmoid per classificazione binaria
        Dense(1, activation='sigmoid')
    ])
    
    # Compila secondo le specifiche
    model.compile(
        loss='binary_crossentropy',    # Per classificazione binaria
        optimizer='adam',              # Adam optimizer
        metrics=['accuracy']           # Accuracy come metrica
    )
    
    # Stampa architettura del modello
    print("\nArchitettura del modello:")
    model.summary()
    
    # Conteggio parametri
    total_params = model.count_params()
    print(f"\nParametri totali: {total_params:,}")
    
    return model

def create_alternative_sonar_model(input_dim: int = 60, hidden_layers: int = 1) -> Sequential:
    """
    Modello alternativo con più layer nascosti per confronto
    
    Args:
        input_dim: Numero di features di input
        hidden_layers: Numero di layer nascosti aggiuntivi
        
    Returns:
        Modello Keras alternativo
    """
    print(f"Creazione modello alternativo con {hidden_layers} layer nascosti...")
    
    model = Sequential([Input(shape=(input_dim,))])
    
    # Layer di input
    model.add(Dense(60, activation='relu'))
    
    # Layer nascosti aggiuntivi
    nodes = 60
    for i in range(hidden_layers):
        nodes = max(10, nodes // 2)  # Dimezza i nodi ad ogni layer
        model.add(Dense(nodes, activation='relu'))
        print(f"Layer nascosto {i+1}: {nodes} nodi")
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def get_model_info() -> dict:
    """
    Restituisce informazioni sul modello sonar
    
    Returns:
        Dizionario con info del modello
    """
    return {
        'architecture': 'Sequential',
        'layers': [
            {'type': 'Input', 'shape': '(60,)'},
            {'type': 'Dense', 'nodes': 60, 'activation': 'relu'},
            {'type': 'Dense', 'nodes': 1, 'activation': 'sigmoid'}
        ],
        'compilation': {
            'loss': 'binary_crossentropy',
            'optimizer': 'adam',
            'metrics': ['accuracy']
        },
        'training_params': {
            'epochs': 100,
            'batch_size': 5
        },
        'task': 'Binary Classification (Metal vs Rock)'
    }