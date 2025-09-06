from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.metrics import precision_score, recall_score
import pandas as pd

# Type annotations
X: NDArray[np.float64]
y: NDArray[np.int32]
X_train: NDArray[np.float64]
X_test: NDArray[np.float64]
y_train: NDArray[np.int32]
y_test: NDArray[np.int32]

# Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Experiment: Different number of epochs")
print("=" * 50)

# Different epochs to test
epochs_list = [5, 10, 20, 50, 100]
results = []

for epochs in epochs_list:
    print(f"\nTesting with {epochs} epochs...")
    
    # Create model
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', 
                 optimizer='adam',
                 metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    # Train model
    history = model.fit(X_train, y_train, 
                       epochs=epochs, 
                       batch_size=32,
                       validation_split=0.2, 
                       verbose=0)
    
    # Make predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Store results
    results.append({
        'epochs': epochs,
        'precision': precision,
        'recall': recall,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1]
    })
    
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")

# Create results DataFrame
df_results = pd.DataFrame(results)
print("\n" + "=" * 50)
print("SUMMARY RESULTS:")
print(df_results.to_string(index=False))

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Precision and Recall plot
ax1.plot(df_results['epochs'], df_results['precision'], 'bo-', label='Precision')
ax1.plot(df_results['epochs'], df_results['recall'], 'ro-', label='Recall')
ax1.set_xlabel('Number of Epochs')
ax1.set_ylabel('Score')
ax1.set_title('Precision and Recall vs Number of Epochs')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(df_results['epochs'], df_results['final_train_loss'], 'go-', label='Train Loss')
ax2.plot(df_results['epochs'], df_results['final_val_loss'], 'mo-', label='Validation Loss')
ax2.set_xlabel('Number of Epochs')
ax2.set_ylabel('Loss')
ax2.set_title('Final Loss vs Number of Epochs')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('epochs_experiment.png', dpi=300, bbox_inches='tight')
print(f"\nGraph saved as 'epochs_experiment.png'")