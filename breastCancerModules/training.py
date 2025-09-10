import tensorflow as tf
from typing import Dict
import numpy as np


def train_and_evaluate(
    model: tf.keras.Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    model_name: str
) -> Dict[str, float]:
    """
    Addestra e valuta un modello

    Args:
        model: Modello Keras da addestrare
        X_train: Dati di training
        y_train: Target di training
        X_test: Dati di test
        y_test: Target di test
        epochs: Numero di epoche
        batch_size: Dimensione del batch
        model_name: Nome del modello per il log

    Returns:
        Dizionario con le metriche finali
    """
    print(
        f"\nAddestrando {model_name} - Epochs: {epochs}, Batch size: {batch_size}")

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    test_loss, test_precision, test_recall = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    f1_score = 2 * (test_precision * test_recall) / (test_precision +
                                                     test_recall) if (test_precision + test_recall) > 0 else 0

    results = {
        'model': model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': f1_score,
        'loss': test_loss
    }

    return results


def run_epochs_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    create_model_func: callable,
    input_shape: int
) -> list:
    """Esegue l'esperimento per testare l'effetto del numero di epochs"""
    print("\n" + "="*60)
    print("ESPERIMENTO 1: Analisi effetto del numero di epochs")
    print("="*60)

    epochs_list = [5, 10, 20, 50, 100]
    epochs_results = []

    for epochs in epochs_list:
        model = create_model_func(input_shape)
        results = train_and_evaluate(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs,
            batch_size=32,
            model_name="Basic Model"
        )
        epochs_results.append(results)

    return epochs_results


def run_batch_size_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    create_model_func,
    input_shape: int
) -> list:
    """Esegue l'esperimento per testare l'effetto della dimensione del batch"""
    print("\n" + "="*60)
    print("ESPERIMENTO 2: Analisi effetto della dimensione del batch")
    print("="*60)

    batch_sizes = [1, 8, 16, 32, 64, 128]
    batch_size_results = []

    for batch_size in batch_sizes:
        model = create_model_func(input_shape)
        results = train_and_evaluate(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=20,
            batch_size=batch_size,
            model_name="Basic Model"
        )
        batch_size_results.append(results)

    return batch_size_results


def run_architecture_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    create_basic_model_func: callable,
    create_funnel_model_func: callable,
    input_shape: int
) -> list:
    """Esegue l'esperimento per confrontare le architetture"""
    print("\n" + "="*60)
    print("ESPERIMENTO 3: Confronto architettura base vs imbuto")
    print("="*60)

    basic_model = create_basic_model_func(input_shape)
    basic_results = train_and_evaluate(
        basic_model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=20,
        batch_size=32,
        model_name="Basic Architecture"
    )

    funnel_model = create_funnel_model_func(input_shape)
    funnel_results = train_and_evaluate(
        funnel_model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=20,
        batch_size=32,
        model_name="Funnel Architecture"
    )

    return [basic_results, funnel_results]


def run_comprehensive_architecture_comparison(
    X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        create_basic_model_func,
        create_funnel_model_func,
        input_shape: int
) -> list:
    """
    Confronto approfondito tra architetture con diverse configurazioni di epochs e batch size
    """
    print("\n" + "="*80)
    print("ESPERIMENTO 4: CONFRONTO APPROFONDITO ARCHITETTURE")
    print("Testando diverse combinazioni di epochs e batch size per entrambe le architetture")
    print("="*80)

    # Configurazioni da testare
    configurations = [
        {"epochs": 20, "batch_size": 32, "name": "Standard"},
        {"epochs": 50, "batch_size": 32, "name": "High Epochs"},
        {"epochs": 100, "batch_size": 32, "name": "Very High Epochs"},
        {"epochs": 20, "batch_size": 64, "name": "Large Batch"},
        {"epochs": 50, "batch_size": 64, "name": "High Epochs + Large Batch"},
        {"epochs": 100, "batch_size": 128, "name": "Maximum Config"}
    ]

    comprehensive_results = []

    for config in configurations:
        print(f"\n--- Testando configurazione: {config['name']} ---")
        print(
            f"Epochs: {config['epochs']}, Batch Size: {config['batch_size']}")

        print(f"\n🔷 Basic Architecture - {config['name']}")
        basic_model = create_basic_model_func(input_shape)
        basic_result = train_and_evaluate(
            basic_model, X_train, y_train, X_test, y_test,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            model_name=f"Basic-{config['name']}"
        )
        basic_result['architecture'] = 'Basic'
        basic_result['config_name'] = config['name']
        comprehensive_results.append(basic_result)

        print(f"\n🔶 Funnel Architecture - {config['name']}")
        funnel_model = create_funnel_model_func(input_shape)
        funnel_result = train_and_evaluate(
            funnel_model, X_train, y_train, X_test, y_test,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            model_name=f"Funnel-{config['name']}"
        )
        funnel_result['architecture'] = 'Funnel'
        funnel_result['config_name'] = config['name']
        comprehensive_results.append(funnel_result)

    return comprehensive_results
