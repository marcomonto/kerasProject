# Importazione delle librerie necessarie
from sklearn.datasets import load_breast_cancer  # Dataset per classificazione binaria di tumori al seno
from sklearn.model_selection import train_test_split  # Funzione per dividere i dati in train/test
import numpy as np  # Libreria per operazioni numeriche e array multidimensionali
from numpy.typing import NDArray  # Type hints per array numpy
import matplotlib.pyplot as plt  # Libreria per creare grafici e visualizzazioni
import tensorflow as tf  # Framework di deep learning
from tensorflow.keras import models  # Moduli per costruire modelli neurali
from tensorflow.keras import layers  # Layer/strati delle reti neurali
from typing import Tuple  # Type hints per tuple

# Annotazioni di tipo per le variabili dei dati
# Queste definizioni servono per il type checking e la documentazione
X: NDArray[np.float64]  # Features (caratteristiche) del dataset - array di float64
y: NDArray[np.int32]    # Target (etichette) del dataset - array di interi (0 o 1)
X_train: NDArray[np.float64]  # Features per il training
X_test: NDArray[np.float64]   # Features per il test
y_train: NDArray[np.int32]    # Etichette per il training
y_test: NDArray[np.int32]     # Etichette per il test

# Caricamento del dataset breast cancer di scikit-learn
# return_X_y=True restituisce direttamente features (X) e target (y) come tuple separate
X, y = load_breast_cancer(return_X_y=True)

# Divisione del dataset in set di training e test
# test_size=0.20: 20% dei dati per il test, 80% per il training
# La funzione mescola automaticamente i dati prima di dividerli per evitare bias
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print("Training set dimensions (train_data):")
print(X_train.shape)  # Stampa le dimensioni: (numero_campioni, numero_features)

# Annotazioni di tipo per le variabili del modello
model: tf.keras.Sequential  # Modello sequenziale (layer impilati uno dopo l'altro)
history: tf.keras.callbacks.History  # Oggetto che registra le metriche durante il training
test_loss: float  # Valore della loss function sul test set
test_pr: float    # Valore della precisione sul test set

# Creazione del modello sequenziale
# Un modello sequenziale è una pila lineare di layer, dove ogni layer ha esattamente
# un tensor di input e un tensor di output
model = models.Sequential()

# PRIMO LAYER (Input Layer)
# Il primo layer deve conoscere le dimensioni di input dei dati
# Dense = layer completamente connesso (ogni neurone è collegato a tutti i neuroni del layer precedente)
# 64 = numero di neuroni/unità in questo layer
# activation='relu' = funzione di attivazione ReLU (Rectified Linear Unit): f(x) = max(0, x)
#   - Aiuta a risolvere il problema del gradiente che svanisce
#   - È computazionalmente efficiente
# input_shape=(X_train.shape[1],) = forma dell'input, dove X_train.shape[1] è il numero di features
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

# SECONDO LAYER (Hidden Layer)
# Layer nascosto con 64 neuroni
# Dopo il primo layer, non è necessario specificare la dimensione di input
# perché Keras la deduce automaticamente dalla forma del layer precedente
# activation='relu' = stessa funzione di attivazione del primo layer per consistency
model.add(layers.Dense(64, activation='relu'))

# TERZO LAYER (Output Layer)
# Layer di output con 1 solo neurone (classificazione binaria)
# activation='sigmoid' = funzione sigmoide: f(x) = 1/(1 + e^(-x))
#   - Schiaccia l'output tra 0 e 1, interpretabile come probabilità
#   - Ideale per classificazione binaria (0 = maligno, 1 = benigno o viceversa)
model.add(layers.Dense(1, activation='sigmoid'))

# COMPILAZIONE DEL MODELLO
# La compilazione configura il processo di apprendimento del modello
model.compile(
    loss='binary_crossentropy',  # Funzione di perdita per classificazione binaria
    #   - Misura quanto le predizioni si discostano dai valori reali
    #   - Binary crossentropy: -[y*log(p) + (1-y)*log(1-p)]
    #   - Penalizza fortemente le predizioni sbagliate con alta confidenza
    
    optimizer='adam',  # Algoritmo di ottimizzazione Adam
    #   - Versione avanzata della discesa del gradiente
    #   - Adatta automaticamente il learning rate per ogni parametro
    #   - Combina i vantaggi di RMSprop e Momentum
    #   - Generalmente più efficace di SGD classico
    
    metrics=[tf.keras.metrics.Precision()]  # Metriche da monitorare durante il training
    #   - Precision = True Positives / (True Positives + False Positives)
    #   - Misura la frazione di predizioni positive che sono effettivamente corrette
    #   - Importante quando il costo dei falsi positivi è alto
)

# TRAINING DEL MODELLO
# Addestramento del modello sui dati di training
history = model.fit(
    X_train, y_train,  # Dati di input e target per il training
    
    epochs=10,  # Numero di epoche (passaggi completi attraverso il dataset)
    #   - Un'epoca = il modello ha visto tutti i campioni di training una volta
    #   - Più epoche permettono al modello di imparare meglio, ma rischio overfitting
    
    batch_size=1,  # Dimensione del batch (numero di campioni processati insieme)
    #   - batch_size=1: aggiorna i pesi dopo ogni singolo campione (Stochastic GD)
    #   - Valori più alti: più stabile ma meno frequenti aggiornamenti
    #   - Compromesso tra stabilità e velocità di convergenza
    
    validation_split=0.2,  # Frazione dei dati di training usata per validazione
    #   - 20% dei dati di training vengono riservati per la validazione
    #   - Utilizzati per monitorare le performance durante il training
    #   - Aiutano a detectare overfitting
    
    verbose=1  # Livello di verbosità (0=silenzioso, 1=progress bar, 2=una linea per epoca)
)

# VALUTAZIONE DEL MODELLO
# Valutazione delle performance sui dati di test (mai visti durante il training)
test_loss, test_pr = model.evaluate(X_test, y_test)
#   - test_loss: valore della loss function sui dati di test
#   - test_pr: valore della precision sui dati di test
#   - Queste metriche indicano quanto bene il modello generalizza su dati nuovi

print(f"Precision sui dati di test: {test_pr:.4f}")  # Stampa la precision con 4 decimali


# VISUALIZZAZIONE DELLE PERFORMANCE
# Creazione di un grafico che mostra l'andamento della loss durante il training
plt.figure()  # Crea una nuova figura per il plot
plt.xlabel('Epoch')  # Etichetta asse X: numero dell'epoca
plt.ylabel('Loss')   # Etichetta asse Y: valore della loss function

# Plot della loss sul training set
plt.plot(
    history.epoch,  # Asse X: numeri delle epoche (0, 1, 2, ..., 9)
    np.array(history.history['loss']),  # Asse Y: valori della loss per il training
    label='Train loss'  # Etichetta per la legenda
)
#   - history.history è un dizionario che contiene le metriche per ogni epoca
#   - 'loss' contiene i valori della loss function calcolati sui dati di training
#   - Tendenza decrescente indica che il modello sta imparando

# Plot della loss sul validation set
plt.plot(
    history.epoch,  # Asse X: stesso delle epoche
    np.array(history.history['val_loss']),  # Asse Y: loss sui dati di validazione
    label='Val loss'  # Etichetta per la legenda
)
#   - 'val_loss' contiene i valori della loss sui dati di validazione
#   - Importante per detectare overfitting:
#     * Se val_loss diminuisce con train_loss → buon apprendimento
#     * Se val_loss aumenta mentre train_loss diminuisce → overfitting

plt.legend()  # Mostra la legenda con le etichette Train/Val loss
plt.savefig('training_loss.png')  # Salva il grafico come file PNG
print("Grafico salvato come 'training_loss.png'")

# SIGNIFICATO TEORICO DEL GRAFICO:
# - Training loss decrescente: il modello migliora sui dati di training
# - Validation loss decrescente: il modello generalizza bene su dati non visti
# - Gap piccolo tra le due curve: buon bilanciamento, nessun overfitting significativo
# - Validation loss che sale mentre training loss scende: segnale di overfitting