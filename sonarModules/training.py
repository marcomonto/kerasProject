import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, List

def train_sonar_model(model: tf.keras.Sequential, 
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      epochs: int = 100, 
                      batch_size: int = 5) -> Tuple[tf.keras.callbacks.History, Dict]:
    """
    Addestra il modello sonar secondo le specifiche
    
    Args:
        model: Modello Keras da addestrare
        X_train: Features di training
        y_train: Labels di training (0/1)
        X_test: Features di test
        y_test: Labels di test (0/1)
        epochs: Numero di epoche (default=100)
        batch_size: Dimensione batch (default=5)
        
    Returns:
        Tuple con history del training e metriche finali
    """
    print(f"Inizio training modello sonar...")
    print(f"Parametri: epochs={epochs}, batch_size={batch_size}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Training del modello
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),  # Usa test set come validation
        verbose=1,
        shuffle=True
    )
    
    # Valutazione finale
    print("\nValutazione finale sul test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Metriche complete
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'final_train_loss': history.history['loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1]
    }
    
    return history, results

def make_predictions_and_evaluate(model: tf.keras.Sequential, 
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 encoder: LabelEncoder) -> Dict:
    """
    Fa predizioni sul test set e valuta il modello con metriche dettagliate
    
    Args:
        model: Modello addestrato
        X_test: Features di test
        y_test: Labels vere di test
        encoder: Encoder per le classi
        
    Returns:
        Dizionario con tutte le metriche di valutazione
    """
    print("\n" + "="*60)
    print("PREDIZIONI E VALUTAZIONE MODELLO")
    print("="*60)
    
    # Predizioni probabilistiche
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Predizioni binarie (soglia 0.5)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    y_pred_binary = y_pred_binary.flatten()
    
    # Calcola metriche
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    
    # Interpretazione delle classi
    class_names = encoder.classes_
    target_names = [f"{cls} ({class_names[i]})" for i, cls in enumerate(['Rock', 'Metal'])]
    report = classification_report(y_test, y_pred_binary, target_names=target_names)
    print(report)
    
    results = {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'predictions_proba': y_pred_proba,
        'predictions_binary': y_pred_binary
    }
    
    return results

def analyze_prediction_confidence(y_pred_proba: np.ndarray, y_test: np.ndarray, 
                                threshold: float = 0.5) -> None:
    """
    Analizza la confidenza delle predizioni
    
    Args:
        y_pred_proba: Probabilità predette
        y_test: Label vere
        threshold: Soglia di decisione
    """
    print(f"\nANALISI CONFIDENZA PREDIZIONI:")
    print("-" * 40)
    
    y_pred_proba_flat = y_pred_proba.flatten()
    
    # Confidenza media per classe
    high_conf_metal = y_pred_proba_flat > 0.8
    high_conf_rock = y_pred_proba_flat < 0.2
    uncertain = (y_pred_proba_flat >= 0.2) & (y_pred_proba_flat <= 0.8)
    
    print(f"Predizioni con alta confidenza Metallo (>0.8): {np.sum(high_conf_metal)}")
    print(f"Predizioni con alta confidenza Roccia (<0.2): {np.sum(high_conf_rock)}")
    print(f"Predizioni incerte (0.2-0.8): {np.sum(uncertain)}")
    
    # Accuracy per livello di confidenza
    if np.sum(high_conf_metal) > 0:
        metal_correct = np.sum((y_pred_proba_flat > 0.8) & (y_test == 1))
        metal_accuracy = metal_correct / np.sum(high_conf_metal)
        print(f"Accuracy su predizioni confident Metallo: {metal_accuracy:.3f}")
    
    if np.sum(high_conf_rock) > 0:
        rock_correct = np.sum((y_pred_proba_flat < 0.2) & (y_test == 0))
        rock_accuracy = rock_correct / np.sum(high_conf_rock)
        print(f"Accuracy su predizioni confident Roccia: {rock_accuracy:.3f}")

def show_sample_predictions(X_test: np.ndarray, y_test: np.ndarray, 
                           y_pred_proba: np.ndarray, encoder: LabelEncoder, 
                           num_samples: int = 10) -> None:
    """
    Mostra predizioni su campioni specifici
    
    Args:
        X_test: Features di test
        y_test: Label vere
        y_pred_proba: Probabilità predette
        encoder: Encoder delle classi
        num_samples: Numero di campioni da mostrare
    """
    print(f"\nCAMPIONI DI PREDIZIONI:")
    print("-" * 70)
    
    # Seleziona campioni casuali usando seed fisso
    rng = np.random.RandomState(42)
    indices = rng.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    print(f"{'Sample':<8} {'True':<8} {'Pred':<8} {'Prob':<8} {'Correct':<8}")
    print("-" * 50)
    
    class_names = encoder.classes_
    
    for i in indices:
        true_class = y_test[i]
        pred_proba = y_pred_proba[i][0]
        pred_class = 1 if pred_proba > 0.5 else 0
        is_correct = "✓" if true_class == pred_class else "✗"
        
        true_name = class_names[true_class]
        pred_name = class_names[pred_class]
        
        print(f"{i:<8} {true_name:<8} {pred_name:<8} {pred_proba:<8.3f} {is_correct:<8}")

def train_improved_sonar_model(model: tf.keras.Sequential, 
                              X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              callbacks: List = None,
                              epochs: int = 100, 
                              batch_size: int = 5) -> Tuple[tf.keras.callbacks.History, Dict]:
    """
    Addestra il modello sonar MIGLIORATO con Early Stopping e Dropout
    
    Args:
        model: Modello Keras da addestrare (con Dropout)
        X_train: Features di training
        y_train: Labels di training (0/1)
        X_test: Features di test
        y_test: Labels di test (0/1)
        callbacks: Lista di callback (es. EarlyStopping)
        epochs: Numero massimo di epoche (default=100)
        batch_size: Dimensione batch (default=5)
        
    Returns:
        Tuple con history del training e metriche finali
    """
    
    # Training del modello con callbacks
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks if callbacks else [],  # Early stopping
        verbose=1,
        shuffle=True
    )
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Informazioni Early Stopping
    if callbacks:
        stopped_epoch = len(history.history['loss'])
        print(f"⏱️  Training fermato all'epoca: {stopped_epoch}")
        if stopped_epoch < epochs:
            print(f"✅ Early Stopping attivato! Risparmiati {epochs - stopped_epoch} epochs")
    
    # Metriche complete
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'epochs_trained': len(history.history['loss']),
        'early_stopped': len(history.history['loss']) < epochs if callbacks else False
    }
    
    return history, results