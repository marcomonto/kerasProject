# Classificazione binaria segnali sonar: Metallo vs Roccia
# Dataset: 60 features sonar -> Classificazione M (Metal) o R (Rock)

from sonarModules.data_handler import load_and_prepare_sonar_data, setup_reproducibility
from sonarModules.model import create_sonar_classification_model, create_improved_sonar_model, get_early_stopping_callback
from sonarModules.training import (train_sonar_model, train_improved_sonar_model, make_predictions_and_evaluate,
                                  analyze_prediction_confidence, show_sample_predictions)
from sonarModules.visualization import (create_training_history_plots, create_classification_analysis)

def main():
    """Funzione principale per classificazione sonar"""
    
    setup_reproducibility()

    # 1. Caricamento e preparazione dati
    X_train, X_test, y_train, y_test, encoder = load_and_prepare_sonar_data()
    
    # 2. Creazione modello secondo specifiche
    model = create_sonar_classification_model(input_dim=60)
    
    # 3. Training secondo specifiche (epochs=100, batch_size=5)
    history, train_results = train_sonar_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=100,
        batch_size=5
    )
    
    # 4. Visualizzazione training (accuracy/loss over epochs)
    create_training_history_plots(history)
    
    # 5. Predizioni e valutazione completa del modello
    eval_results = make_predictions_and_evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        encoder=encoder
    )
    
    # 6. Analisi confidenza predizioni
    analyze_prediction_confidence(
        y_pred_proba=eval_results['predictions_proba'],
        y_test=y_test
    )
    
    # 7. Esempi di predizioni
    show_sample_predictions(
        X_test=X_test,
        y_test=y_test,
        y_pred_proba=eval_results['predictions_proba'],
        encoder=encoder,
        num_samples=15
    )
    
    # 8. Visualizzazioni complete
    create_classification_analysis(eval_results, encoder)

def main_improved():
    """Funzione per testare il modello MIGLIORATO con Dropout e Early Stopping"""
    
    setup_reproducibility()

    # 1. Caricamento e preparazione dati
    X_train, X_test, y_train, y_test, encoder = load_and_prepare_sonar_data()
    
    # 2. Creazione modello MIGLIORATO con Dropout
    model_improved = create_improved_sonar_model(input_dim=60)
    
    # 3. Setup Early Stopping
    early_stopping = get_early_stopping_callback(patience=15)
    
    # 4. Training con Early Stopping
    history_improved, train_results_improved = train_improved_sonar_model(
        model=model_improved,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        callbacks=[early_stopping],  # Early Stopping attivo
        epochs=100,
        batch_size=5
    )
    
    create_training_history_plots(history_improved, output_suffix="_improved")
    
    # 6. Predizioni e valutazione del modello migliorato
    eval_results_improved = make_predictions_and_evaluate(
        model=model_improved,
        X_test=X_test,
        y_test=y_test,
        encoder=encoder
    )
    
    # 7. Visualizzazioni complete modello migliorato
    create_classification_analysis(eval_results_improved, encoder, output_suffix="_improved")
    

if __name__ == "__main__":
    # Esegui entrambi i modelli per confronto
    main()
    main_improved()