# Moduli per sonar classification
# Package per classificazione binaria segnali sonar (Metal vs Rock)

from .data_handler import (load_and_prepare_sonar_data, get_sonar_dataset_info, 
                          apply_feature_scaling)
from .model import create_sonar_classification_model, create_improved_sonar_model, get_early_stopping_callback
from .training import (train_sonar_model, train_improved_sonar_model, make_predictions_and_evaluate, 
                      analyze_prediction_confidence, show_sample_predictions)
from .visualization import (create_training_history_plots, create_classification_analysis)

__all__ = [
    'load_and_prepare_sonar_data',
    'get_sonar_dataset_info',
    'apply_feature_scaling',
    'create_sonar_classification_model',
    'create_improved_sonar_model',
    'get_early_stopping_callback',
    'train_sonar_model',
    'train_improved_sonar_model',
    'make_predictions_and_evaluate',
    'analyze_prediction_confidence',
    'show_sample_predictions',
    'create_training_history_plots',
    'create_classification_analysis',
]