import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict
import tensorflow as tf

def create_mse_training_plot(history: tf.keras.callbacks.History) -> None:
    """
    Crea il grafico MSE over epochs per training e validation set
    
    Args:
        history: History object dal training del modello
    """
    # Crea directory per i risultati
    output_dir = "results/cars_grafici"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREAZIONE GRAFICI MSE TRAINING")
    print("="*60)
    
    plt.figure(figsize=(12, 5))
    
    # Grafico 1: MSE over epochs
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(history.history['mse']) + 1)
    
    plt.plot(epochs_range, history.history['mse'], 'b-', label='Training MSE', linewidth=2)
    plt.plot(epochs_range, history.history['val_mse'], 'r-', label='Validation MSE', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Evidenzia il punto di minimo validation MSE
    min_val_mse_epoch = np.argmin(history.history['val_mse']) + 1
    min_val_mse = min(history.history['val_mse'])
    plt.axvline(x=min_val_mse_epoch, color='orange', linestyle='--', alpha=0.7)
    plt.annotate(f'Min Val MSE\nEpoch {min_val_mse_epoch}\nMSE: {min_val_mse:.2f}', 
                xy=(min_val_mse_epoch, min_val_mse), 
                xytext=(min_val_mse_epoch + 20, min_val_mse + (max(history.history['val_mse']) - min_val_mse) * 0.3),
                arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                fontsize=10, ha='center')
    
    # Grafico 2: Loss over epochs (stesso di MSE per regressione)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], 'g-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, history.history['val_loss'], 'm-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mse_training_validation.png', dpi=300, bbox_inches='tight')
    print(f"Grafico MSE salvato: '{output_dir}/mse_training_validation.png'")
    
    # Statistiche finali
    final_train_mse = history.history['mse'][-1]
    final_val_mse = history.history['val_mse'][-1]
    
    print(f"\nStatistiche Training:")
    print(f"- MSE finale training: {final_train_mse:.4f}")
    print(f"- MSE finale validation: {final_val_mse:.4f}")
    print(f"- Differenza: {abs(final_train_mse - final_val_mse):.4f}")
    
    if final_val_mse > final_train_mse * 1.1:
        print("⚠️  Possibile overfitting rilevato")
    else:
        print("✅ Buona generalizzazione")

def create_comprehensive_analysis(history: tf.keras.callbacks.History, results: Dict, 
                                 model: tf.keras.Sequential, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Crea un'analisi completa con multiple visualizzazioni
    
    Args:
        history: History del training
        results: Risultati del modello
        model: Modello addestrato
        X_test: Dati di test
        y_test: Target di test
    """
    output_dir = "results/cars_grafici"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREAZIONE ANALISI COMPLETA")
    print("="*60)
    
    # Predizioni per analisi
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Figura con 4 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: Predizioni vs Valori Reali
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    
    # Linea perfetta (y=x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0, 0].set_xlabel('Actual Car Price ($)')
    axes[0, 0].set_ylabel('Predicted Car Price ($)')
    axes[0, 0].set_title('Predictions vs Actual Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calcola R²
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = ${results["test_rmse"]:.2f}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residui
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Car Price ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribuzione degli errori
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuals ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistiche residui
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    axes[1, 0].text(0.05, 0.95, f'Mean: ${mean_residual:.2f}\nStd: ${std_residual:.2f}', 
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 4. Q-Q plot semplificato (distribuzione dei residui)
    sorted_residuals = np.sort(residuals)
    rng = np.random.RandomState(42)
    normal_quantiles = rng.normal(0, std_residual, len(residuals))
    normal_quantiles = np.sort(normal_quantiles)
    
    axes[1, 1].scatter(normal_quantiles, sorted_residuals, alpha=0.6, color='orange')
    axes[1, 1].plot([normal_quantiles.min(), normal_quantiles.max()], 
                   [sorted_residuals.min(), sorted_residuals.max()], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('Theoretical Normal Quantiles')
    axes[1, 1].set_ylabel('Sample Quantiles')
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Analisi completa salvata: '{output_dir}/comprehensive_analysis.png'")

def create_feature_importance_plot(feature_names: list, importance_scores: list) -> None:
    """
    Crea un grafico per l'importanza delle features
    
    Args:
        feature_names: Nomi delle features
        importance_scores: Score di importanza
    """
    output_dir = "results/cars_grafici"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Ordina per importanza
    sorted_idx = np.argsort(importance_scores)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = [importance_scores[i] for i in sorted_idx]
    
    # Grafico a barre orizzontali
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
    bars = plt.barh(sorted_features, sorted_scores, color=colors, alpha=0.8)
    
    plt.xlabel('Importance Score')
    plt.title('Feature Importance Analysis')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Aggiungi valori sulle barre
    for bar, score in zip(bars, sorted_scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"Importanza features salvata: '{output_dir}/feature_importance.png'")

def print_final_summary(results: Dict) -> None:
    """
    Stampa un riassunto finale dei risultati
    
    Args:
        results: Dizionario con i risultati del modello
    """
    print("\n" + "="*80)
    print("RIASSUNTO FINALE MODELLO CARS REGRESSION")
    print("="*80)
    
    print(f"📊 METRICHE PRINCIPALI:")
    print(f"   • Test MSE: {results['test_mse']:.2f}")
    print(f"   • Test RMSE: ${results['test_rmse']:.2f}")
    print(f"   • Media prezzi reali: ${results['mean_target']:.2f}")
    print(f"   • RMSE come % della media: {results['rmse_percentage']:.1f}%")
    
    print(f"\n🎯 INTERPRETAZIONE:")
    if results['rmse_percentage'] < 10:
        print("   ✅ Eccellente: Errore molto basso")
    elif results['rmse_percentage'] < 20:
        print("   ✅ Buono: Errore accettabile")  
    elif results['rmse_percentage'] < 30:
        print("   ⚠️  Discreto: Errore moderato")
    else:
        print("   ❌ Scarso: Errore elevato")
    
    print(f"\n📈 GRAFICI GENERATI:")
    print(f"   1. mse_training_validation.png - MSE durante training")
    print(f"   2. comprehensive_analysis.png - Analisi predizioni completa")
    print(f"   3. feature_importance.png - Importanza delle features")
    
    print("="*80)