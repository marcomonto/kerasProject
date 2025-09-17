from breastCancerModules.data_handler import setup_reproducibility, load_and_prepare_data
from breastCancerModules.models import create_basic_model, create_funnel_model
from breastCancerModules.training import (run_epochs_experiment, run_batch_size_experiment,
                                          run_architecture_experiment, run_comprehensive_architecture_comparison)
from breastCancerModules.visualization import (create_visualizations,
                                               create_comprehensive_architecture_visualization)


def main():
    """Funzione principale che coordina tutti gli esperimenti"""
    setup_reproducibility()

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    input_shape = X_train.shape[1]

    epochs_results = run_epochs_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        create_basic_model,
        input_shape
    )

    batch_size_results = run_batch_size_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        create_basic_model,
        input_shape
    )

    architecture_results = run_architecture_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        create_basic_model,
        create_funnel_model,
        input_shape
    )

    create_visualizations(
        epochs_results,
        batch_size_results,
        architecture_results
    )

    comprehensive_results = run_comprehensive_architecture_comparison(
        X_train,
        y_train,
        X_test,
        y_test,
        create_basic_model,
        create_funnel_model,
        input_shape
    )

    create_comprehensive_architecture_visualization(comprehensive_results)

    print("\n" + "="*80)
    print("TUTTI GLI ESPERIMENTI COMPLETATI!")
    print("✅ 3 esperimenti standard + 1 confronto approfondito")
    print("✅ 6 grafici dettagliati generati in results/grafici/")
    print("="*80)


if __name__ == "__main__":
    main()
