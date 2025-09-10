# Classificazione binaria segnali sonar: Metallo vs Roccia
# Dataset: 60 features sonar -> Classificazione M (Metal) o R (Rock)

from sonarModules.data_handler import load_and_prepare_sonar_data, get_sonar_dataset_info, setup_reproducibility
from sonarModules.model import create_sonar_classification_model, get_model_info
from sonarModules.training import (train_sonar_model, make_predictions_and_evaluate,
                                  analyze_prediction_confidence, show_sample_predictions)
from sonarModules.visualization import (create_training_history_plots, create_classification_analysis,
                                       create_prediction_confidence_plot, print_final_sonar_summary)

def main():
    """Funzione principale per classificazione sonar"""
    
    setup_reproducibility()
    
    print("🔊 AVVIO SONAR CLASSIFICATION ANALYSIS")
    print("="*80)
    print("Task: Discriminare segnali sonar riflessi da cilindro metallico vs roccia")
    print("Input: 60 features numeriche da segnali sonar")
    print("Output: Classe binaria - M (Metal) o R (Rock)")
    print("="*80)
    
    # 1. Caricamento e preparazione dati
    X_train, X_test, y_train, y_test, encoder = load_and_prepare_sonar_data()
    dataset_info = get_sonar_dataset_info()
    model_info = get_model_info()
    
    print(f"\nDataset info: {dataset_info['description']}")
    print(f"Classi: {dataset_info['classes']} -> {list(dataset_info['class_names'].values())}")
    
    # 2. Creazione modello secondo specifiche
    model = create_sonar_classification_model(input_dim=60)
    
    print(f"\nModello configurato:")
    print(f"Architecture: {model_info['layers']}")
    print(f"Loss: {model_info['compilation']['loss']}")
    print(f"Optimizer: {model_info['compilation']['optimizer']}")
    print(f"Metrics: {model_info['compilation']['metrics']}")
    
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
    create_prediction_confidence_plot(eval_results)
    
    # 9. Riassunto finale
    print_final_sonar_summary(train_results, eval_results)
    
    print(f"\n🎯 OBIETTIVI COMPLETATI:")
    print(f"✅ Modello Sequential: loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']")
    print(f"✅ Training: epochs=100, batch_size=5")
    print(f"✅ Input Dense layer: 60 nodi (input_dim=60, activation='relu')")
    print(f"✅ Output Dense layer: 1 nodo (activation='sigmoid')")
    print(f"✅ Predizioni test set completate e modello valutato")
    print(f"✅ Accuracy finale: {train_results['test_accuracy']:.4f} ({train_results['test_accuracy']*100:.2f}%)")
    print(f"✅ Grafici salvati in results/sonar_grafici/")
    
    # Test specifico su predizioni
    print(f"\n🧪 TEST PREDIZIONI:")
    test_sample = X_test[0:3]
    predictions = model.predict(test_sample, verbose=0)
    print(f"Esempio 3 predizioni: {predictions.flatten()}")
    binary_preds = (predictions > 0.5).astype(int).flatten()
    print(f"Decisioni binarie: {binary_preds}")
    print(f"Classi predette: {encoder.inverse_transform(binary_preds)}")

if __name__ == "__main__":
    main()