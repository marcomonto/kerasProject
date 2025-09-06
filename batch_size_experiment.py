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

print("Experiment: Different batch sizes")
print("=" * 50)

# Different batch sizes to test
batch_sizes = [1, 8, 16, 32, 64, 128]
results = []

for batch_size in batch_sizes:
    print(f"\nTesting with batch_size={batch_size}...")
    
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
                       epochs=20, 
                       batch_size=batch_size,
                       validation_split=0.2, 
                       verbose=0)
    
    # Make predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Calculate training time info (steps per epoch)
    train_samples = int(len(X_train) * 0.8)  # 80% for training after validation split
    steps_per_epoch = train_samples // batch_size + (1 if train_samples % batch_size else 0)
    
    # Store results
    results.append({
        'batch_size': batch_size,
        'precision': precision,
        'recall': recall,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'steps_per_epoch': steps_per_epoch
    })
    
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Steps per epoch: {steps_per_epoch}")

# Create results DataFrame
df_results = pd.DataFrame(results)
print("\n" + "=" * 50)
print("SUMMARY RESULTS:")
print(df_results.to_string(index=False))

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Precision and Recall plot
ax1.semilogx(df_results['batch_size'], df_results['precision'], 'bo-', label='Precision')
ax1.semilogx(df_results['batch_size'], df_results['recall'], 'ro-', label='Recall')
ax1.set_xlabel('Batch Size (log scale)')
ax1.set_ylabel('Score')
ax1.set_title('Precision and Recall vs Batch Size')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.semilogx(df_results['batch_size'], df_results['final_train_loss'], 'go-', label='Train Loss')
ax2.semilogx(df_results['batch_size'], df_results['final_val_loss'], 'mo-', label='Validation Loss')
ax2.set_xlabel('Batch Size (log scale)')
ax2.set_ylabel('Loss')
ax2.set_title('Final Loss vs Batch Size')
ax2.legend()
ax2.grid(True)

# Steps per epoch
ax3.semilogx(df_results['batch_size'], df_results['steps_per_epoch'], 'co-')
ax3.set_xlabel('Batch Size (log scale)')
ax3.set_ylabel('Steps per Epoch')
ax3.set_title('Training Steps per Epoch vs Batch Size')
ax3.grid(True)

# Combined metrics
ax4.plot(df_results['batch_size'], df_results['precision'], 'bo-', label='Precision')
ax4.plot(df_results['batch_size'], df_results['recall'], 'ro-', label='Recall')
ax4.set_xlabel('Batch Size (linear scale)')
ax4.set_ylabel('Score')
ax4.set_title('Precision and Recall vs Batch Size (Linear Scale)')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('batch_size_experiment.png', dpi=300, bbox_inches='tight')
print(f"\nGraph saved as 'batch_size_experiment.png'")