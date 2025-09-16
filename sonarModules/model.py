import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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
        Input(shape=(input_dim,)),
        Dense(60, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',    # Per classificazione binaria
        optimizer='adam',              # Adam optimizer
        metrics=['accuracy']           # Accuracy come metrica
    )
    
    model.summary()
    
    return model

def create_improved_sonar_model(input_dim: int = 60) -> Sequential:
    """
    Crea un modello migliorato con Dropout per prevenire overfitting
    
    Miglioramenti:
    - Aggiunge layer Dropout per regolarizzazione
    - Architettura più profonda per migliore apprendimento
    - Stesso loss e optimizer del modello originale
    
    Architettura:
    - Input Dense layer: 64 nodi (relu) + Dropout(0.3)
    - Hidden Dense layer: 32 nodi (relu) + Dropout(0.5) 
    - Output Dense layer: 1 nodo (sigmoid)
    
    Args:
        input_dim: Numero di features di input (default=60)
        
    Returns:
        Modello Keras compilato con Dropout
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),                    
        Dense(32, activation='relu'),
        Dropout(0.5),                    
        
        Dense(1, activation='sigmoid')  
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

def get_early_stopping_callback(patience: int = 15) -> EarlyStopping:
    """
    Crea callback Early Stopping per prevenire overfitting
    
    Args:
        patience: Numero di epochs senza miglioramento prima di fermarsi
        
    Returns:
        EarlyStopping callback configurato
    """
    return EarlyStopping(
        monitor='val_accuracy',          # Monitora validation accuracy
        mode='max',                      # Cerca il massimo
        patience=patience,               # Aspetta N epochs
        restore_best_weights=True,       # Ripristina i migliori pesi
        verbose=1                        # Stampa quando si ferma
    )