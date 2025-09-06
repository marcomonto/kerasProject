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
import math

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

print("Experiment: Funnel Architecture")
print("=" * 50)
print(f"Input features: {X_train.shape[1]}")

def create_funnel_layers(input_size: int, max_layers: int = 8):
    """
    Create funnel architecture layers based on the requirements:
    - First hidden layer: 75% of input
    - Second: 50% of input  
    - Third: 25% of input
    - Fourth: 12.5% of input
    - Fifth: 6% of input
    - And so on...
    """
    layers_info = []
    percentages = [0.75, 0.50, 0.25, 0.125, 0.06]
    
    for i, percentage in enumerate(percentages):
        neurons = max(1, int(input_size * percentage))
        layers_info.append({
            'layer': i + 1,
            'neurons': neurons,
            'percentage': percentage * 100
        })
        if len(layers_info) >= max_layers:
            break
    
    # Continue with halving pattern if needed
    if len(layers_info) < max_layers:
        current_neurons = layers_info[-1]['neurons']
        for i in range(len(layers_info), max_layers):
            current_neurons = max(1, current_neurons // 2)
            if current_neurons <= 1:
                break
            layers_info.append({
                'layer': i + 1,
                'neurons': current_neurons,
                'percentage': (current_neurons / input_size) * 100
            })
    
    return layers_info

# Calculate funnel architecture
input_size = X_train.shape[1]
funnel_info = create_funnel_layers(input_size)

print("\nFunnel Architecture Design:")
print("Layer | Neurons | % of Input")
print("-" * 30)
for layer_info in funnel_info:
    print(f"{layer_info['layer']:5d} | {layer_info['neurons']:7d} | {layer_info['percentage']:8.1f}%")

# Test different numbers of layers
layer_configs = [
    {"name": "Original (2 layers)", "layers": [64, 64]},
    {"name": "Funnel 3 layers", "layers": [funnel_info[0]['neurons'], funnel_info[1]['neurons'], funnel_info[2]['neurons']]},
    {"name": "Funnel 4 layers", "layers": [layer['neurons'] for layer in funnel_info[:4]]},
    {"name": "Funnel 5 layers", "layers": [layer['neurons'] for layer in funnel_info[:5]]},
]

# Add full funnel if we have enough layers
if len(funnel_info) > 5:
    layer_configs.append({
        "name": f"Funnel {len(funnel_info)} layers", 
        "layers": [layer['neurons'] for layer in funnel_info]
    })

results = []

for config in layer_configs:
    print(f"\n{'='*20}")
    print(f"Testing: {config['name']}")
    print(f"Architecture: {config['layers']}")
    print(f"Total parameters estimate: {sum(config['layers']) + len(config['layers'])}")
    
    # Create model
    model = models.Sequential()
    
    # First layer (input layer)
    model.add(layers.Dense(config['layers'][0], 
                          activation='relu', 
                          input_shape=(X_train.shape[1],)))
    
    # Hidden layers
    for neurons in config['layers'][1:]:
        model.add(layers.Dense(neurons, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', 
                 optimizer='adam',
                 metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    # Train model
    history = model.fit(X_train, y_train, 
                       epochs=30, 
                       batch_size=32,
                       validation_split=0.2, 
                       verbose=0)
    
    # Make predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Count parameters
    total_params = model.count_params()
    
    # Store results
    results.append({
        'architecture': config['name'],
        'layers': str(config['layers']),
        'num_layers': len(config['layers']),
        'total_neurons': sum(config['layers']),
        'total_params': total_params,
        'precision': precision,
        'recall': recall,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    })
    
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {results[-1]['f1_score']:.4f}")
    print(f"  Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Total parameters: {total_params}")

# Create results DataFrame
df_results = pd.DataFrame(results)
print("\n" + "=" * 80)
print("SUMMARY RESULTS:")
print(df_results[['architecture', 'num_layers', 'total_neurons', 'precision', 'recall', 'f1_score']].to_string(index=False))

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Precision and Recall vs Number of Layers
ax1.plot(df_results['num_layers'], df_results['precision'], 'bo-', label='Precision', markersize=8)
ax1.plot(df_results['num_layers'], df_results['recall'], 'ro-', label='Recall', markersize=8)
ax1.plot(df_results['num_layers'], df_results['f1_score'], 'go-', label='F1-Score', markersize=8)
ax1.set_xlabel('Number of Hidden Layers')
ax1.set_ylabel('Score')
ax1.set_title('Metrics vs Number of Hidden Layers')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(df_results['num_layers'])

# Loss vs Number of Layers
ax2.plot(df_results['num_layers'], df_results['final_train_loss'], 'go-', label='Train Loss', markersize=8)
ax2.plot(df_results['num_layers'], df_results['final_val_loss'], 'mo-', label='Validation Loss', markersize=8)
ax2.set_xlabel('Number of Hidden Layers')
ax2.set_ylabel('Loss')
ax2.set_title('Final Loss vs Number of Hidden Layers')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(df_results['num_layers'])

# Total Parameters vs Performance
ax3.scatter(df_results['total_params'], df_results['precision'], s=100, alpha=0.7, label='Precision')
ax3.scatter(df_results['total_params'], df_results['recall'], s=100, alpha=0.7, label='Recall')
ax3.scatter(df_results['total_params'], df_results['f1_score'], s=100, alpha=0.7, label='F1-Score')
ax3.set_xlabel('Total Parameters')
ax3.set_ylabel('Score')
ax3.set_title('Performance vs Model Complexity')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Architecture comparison (bar plot)
x_pos = np.arange(len(df_results))
width = 0.25
ax4.bar(x_pos - width, df_results['precision'], width, label='Precision', alpha=0.8)
ax4.bar(x_pos, df_results['recall'], width, label='Recall', alpha=0.8)
ax4.bar(x_pos + width, df_results['f1_score'], width, label='F1-Score', alpha=0.8)
ax4.set_xlabel('Architecture')
ax4.set_ylabel('Score')
ax4.set_title('Performance Comparison by Architecture')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([arch.split()[0] for arch in df_results['architecture']], rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('funnel_architecture_experiment.png', dpi=300, bbox_inches='tight')
print(f"\nGraph saved as 'funnel_architecture_experiment.png'")

# Print best performing architecture
best_f1_idx = df_results['f1_score'].idxmax()
best_arch = df_results.iloc[best_f1_idx]
print(f"\nBest performing architecture (F1-Score):")
print(f"  {best_arch['architecture']}")
print(f"  F1-Score: {best_arch['f1_score']:.4f}")
print(f"  Precision: {best_arch['precision']:.4f}")
print(f"  Recall: {best_arch['recall']:.4f}")
print(f"  Total parameters: {best_arch['total_params']}")