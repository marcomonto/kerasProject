import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict
import tensorflow as tf

def create_training_history_plots(history: tf.keras.callbacks.History, output_suffix: str = "") -> None:
    """
    Crea grafici per accuracy e loss durante il training
    
    Args:
        history: History object dal training
        output_suffix: Suffisso per il nome del file (es. "_improved")
    """
    output_dir = "results/sonar_grafici"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREAZIONE GRAFICI TRAINING HISTORY")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    # Grafico 1: Accuracy
    axes[0].plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Evidenzia best validation accuracy
    best_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    axes[0].axvline(x=best_val_acc_epoch, color='orange', linestyle='--', alpha=0.7)
    axes[0].annotate(f'Best Val Acc\nEpoch {best_val_acc_epoch}\nAcc: {best_val_acc:.3f}', 
                    xy=(best_val_acc_epoch, best_val_acc), 
                    xytext=(best_val_acc_epoch + 10, best_val_acc - 0.1),
                    arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                    fontsize=10, ha='center')
    
    # Grafico 2: Loss
    axes[1].plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history{output_suffix}.png', dpi=300, bbox_inches='tight')
    print(f"Grafici training salvati: '{output_dir}/training_history{output_suffix}.png'")
    
    # Statistiche finali
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nStatistiche Finali:")
    print(f"Training   - Accuracy: {final_train_acc:.4f}, Loss: {final_train_loss:.4f}")
    print(f"Validation - Accuracy: {final_val_acc:.4f}, Loss: {final_val_loss:.4f}")
    
    if final_val_acc > final_train_acc * 0.95:
        print("✅ Buona generalizzazione")
    else:
        print("⚠️  Possibile overfitting")

def create_classification_analysis(eval_results: Dict, encoder, output_suffix: str = "") -> None:
    """
    Crea analisi complete per classificazione binaria
    
    Args:
        eval_results: Risultati della valutazione
        encoder: Label encoder
        output_suffix: Suffisso per il nome del file (es. "_improved")
    """
    output_dir = "results/sonar_grafici"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = eval_results['confusion_matrix']
    class_names = ['Rock (R)', 'Metal (M)']
    
    # Crea heatmap manualmente senza seaborn
    im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    
    # Aggiungi etichette
    axes[0, 0].set_xticks(range(len(class_names)))
    axes[0, 0].set_yticks(range(len(class_names)))
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 0].set_yticklabels(class_names)
    
    # Aggiungi valori nella matrice
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center', 
                           color='white' if cm[i, j] > cm.max()/2 else 'black', 
                           fontsize=14, fontweight='bold')
    
    # 2. Metriche per classe
    metrics = ['Precision', 'Recall', 'Specificity', 'F1-Score']
    values = [eval_results['precision'], eval_results['recall'], 
              eval_results['specificity'], eval_results['f1_score']]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[0, 1].set_title('Classification Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori sulle barre
    for bar, value in zip(bars, values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Rimuovi il terzo subplot (Distribution of Predicted Probabilities)
    axes[1, 0].set_visible(False)
    
    # Rimuovi il quarto subplot (ROC Curve)
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/classification_analysis{output_suffix}.png', dpi=300, bbox_inches='tight')