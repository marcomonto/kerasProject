# Neural Networks Theory Summary - ML2025 Lab5

## Reti Neurali Artificiali (ANN)

### Definizione
Le reti neurali artificiali sono modelli computazionali ispirati al funzionamento del cervello umano, costituiti da:
- **Neuroni artificiali** (nodi): unità di calcolo che processano informazioni
- **Connessioni weighted**: links tra neuroni con pesi che determinano l'intensità del segnale
- **Architettura a livelli**: input layer, hidden layer(s), output layer

### Componenti Fondamentali

#### 1. Neuroni e Attivazioni
Ogni neurone calcola:
```
output = activation_function(Σ(weights * inputs) + bias)
```

#### 2. Funzioni di Attivazione
- **Linear**: `f(x) = x` - per regressione (output continuo)
- **Sigmoid**: `f(x) = 1/(1+e^-x)` - per classificazione binaria (0-1)
- **ReLU**: `f(x) = max(0,x)` - più efficiente, previene vanishing gradient
- **Tanh**: `f(x) = tanh(x)` - output tra -1 e 1

#### 3. Loss Functions
- **Mean Squared Error (MSE)**: per regressione
  ```
  MSE = (1/n) * Σ(y_true - y_pred)²
  ```
- **Binary Crossentropy**: per classificazione binaria
  ```
  BCE = -[y*log(p) + (1-y)*log(1-p)]
  ```
- **Categorical Crossentropy**: per classificazione multi-classe

#### 4. Ottimizzatori
- **Adam**: adattivo, combina momentum e RMSprop
- **SGD**: Stochastic Gradient Descent classico
- **RMSprop**: adatta learning rate per ogni parametro

## Hyperparameters Critici

### 1. Epochs
- Numero di passate complete attraverso il training set
- **Troppo pochi**: underfitting
- **Troppi**: overfitting

### 2. Batch Size
- Numero di campioni processati prima di aggiornare i pesi
- **Piccolo** (5-32): aggiornamenti più frequenti, training più noisy ma generalizza meglio
- **Grande** (128-512): training più stabile ma può essere meno generalizzabile

### 3. Learning Rate
- Velocità di apprendimento del modello
- **Alto**: convergenza veloce ma rischio di instabilità
- **Basso**: convergenza lenta ma più stabile

### 4. Architettura
- **Layers**: più layer = maggiore capacità di apprendimento complesso
- **Nodes per layer**: più nodi = maggiore capacità ma rischio overfitting
- **Funnel Architecture**: riduzione progressiva dei nodi per feature extraction

## Tecniche di Regolarizzazione

### 1. Validation Split
- Separazione dei dati in training/validation/test
- Monitoraggio performance su validation per early stopping

### 2. Early Stopping
- Arresto training quando validation loss smette di migliorare
- Previene overfitting mantenendo generalizzazione

### 3. Dropout
- Disattivazione casuale di neuroni durante training
- Riduce overfitting e migliora generalizzazione

## Keras Implementation Pattern

```python
# 1. Creazione modello
model = Sequential([
    Input(shape=(input_features,)),
    Dense(nodes, activation='relu'),
    Dense(output_size, activation='sigmoid/linear')
])

# 2. Compilazione
model.compile(
    loss='binary_crossentropy/mse',
    optimizer='adam',
    metrics=['accuracy/mse']
)

# 3. Training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val)
)
```

## Tipi di Problemi ML

### 1. Classificazione Binaria
- **Output**: probabilità (0-1)
- **Loss**: binary_crossentropy  
- **Activation finale**: sigmoid
- **Metriche**: accuracy, precision, recall, F1-score

### 2. Regressione
- **Output**: valore continuo
- **Loss**: MSE o MAE
- **Activation finale**: linear (nessuna)
- **Metriche**: MSE, RMSE, MAE

### 3. Classificazione Multi-classe
- **Output**: probabilità per ogni classe
- **Loss**: categorical_crossentropy
- **Activation finale**: softmax
- **Metriche**: accuracy, precision/recall per classe

## Evaluation Metrics

### Classificazione
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN)
- **Precision**: TP/(TP+FP)
- **Recall**: TP/(TP+FN)  
- **F1-Score**: 2*(Precision*Recall)/(Precision+Recall)
- **Confusion Matrix**: visualizzazione errori per classe

### Regressione
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficiente di determinazione

## Best Practices

1. **Data Preprocessing**
   - Normalizzazione/standardizzazione features
   - Encoding variabili categoriche
   - Train/validation/test split stratificato

2. **Model Design**
   - Iniziare con architettura semplice
   - Aggiungere complessità gradualmente
   - Usare ReLU per hidden layers

3. **Training Strategy**
   - Monitorare training e validation loss
   - Implementare early stopping
   - Salvare best model

4. **Hyperparameter Tuning**
   - Grid search o random search
   - Cross-validation per robustezza
   - Validazione su test set finale

---

## Collegamento con gli Esercizi Implementati

### Esercizio 1: Breast Cancer Classification
**Tipo**: Classificazione binaria (Maligno/Benigno)
- **Architettura**: Sequential con funnel design (64→32→1)
- **Loss**: `binary_crossentropy` (appropriata per output binario)
- **Activation**: ReLU per hidden layers, Sigmoid per output
- **Metriche**: accuracy, precision, recall, F1-score, confusion matrix
- **Dataset**: 569 campioni, 30 features mediche
- **Confronto**: Architettura base vs funnel per ottimizzazione performance

### Esercizio 2: Cars Regression  
**Tipo**: Regressione (previsione prezzo auto)
- **Architettura**: Sequential (64→32→1) senza activation finale
- **Loss**: `MSE` (Mean Squared Error per valori continui)
- **Activation**: ReLU per hidden layers, Linear per output
- **Metriche**: MSE, RMSE per valutazione errore predizione
- **Dataset**: Features multiple (marca, anno, km, etc.) → prezzo
- **Focus**: Predizione accurata di valori continui

### Esercizio 3: Sonar Classification
**Tipo**: Classificazione binaria (Metal/Rock)
- **Architettura**: Sequential semplice (60→1)
- **Loss**: `binary_crossentropy` per discriminazione segnali
- **Activation**: ReLU per input layer, Sigmoid per decisione finale
- **Metriche**: accuracy, precision, recall, specificity, confusion matrix
- **Dataset**: 208 campioni, 60 features sonar processing
- **Parametri**: epochs=100, batch_size=5 per training preciso

### Pattern Comuni Implementati

1. **Modularizzazione**: 
   - `data_handler.py`: preprocessing e data loading
   - `model.py`: architetture neurali
   - `training.py`: loop di training e evaluation
   - `visualization.py`: grafici e analisi risultati

2. **Reproducibility**: seed random per risultati consistenti

3. **Validation Strategy**: split 80/20 con monitoraggio val_loss

4. **Comprehensive Analysis**: 
   - Training history plots (accuracy/loss over epochs)
   - Classification analysis (confusion matrix, metriche per classe)  
   - Confidence analysis (distribuzione probabilità predette)

### Insights Teorici Applicati

- **ReLU**: usata in tutti hidden layers per efficienza computazionale
- **Sigmoid**: usata per classificazione binaria (output 0-1)
- **Linear**: usata per regressione (output senza vincoli)
- **Adam Optimizer**: scelto per convergenza adattiva efficace
- **Batch Size Piccolo**: 5-32 per miglior generalizzazione
- **Early Stopping**: implementato via validation monitoring
- **Feature Scaling**: standardizzazione per convergenza stabile