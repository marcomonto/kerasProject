# Moduli per sonar classification
# Package per classificazione binaria segnali sonar (Metal vs Rock)

from .data_handler import (load_and_prepare_sonar_data, get_sonar_dataset_info, 
                          apply_feature_scaling)
from .model import create_sonar_classification_model, create_alternative_sonar_model, get_model_info
from .training import (train_sonar_model, make_predictions_and_evaluate, 
                      analyze_prediction_confidence, show_sample_predictions)
from .visualization import (create_training_history_plots, create_classification_analysis,
                           create_prediction_confidence_plot, print_final_sonar_summary)

__all__ = [
    'load_and_prepare_sonar_data',
    'get_sonar_dataset_info',
    'apply_feature_scaling',
    'create_sonar_classification_model',
    'create_alternative_sonar_model', 
    'get_model_info',
    'train_sonar_model',
    'make_predictions_and_evaluate',
    'analyze_prediction_confidence',
    'show_sample_predictions',
    'create_training_history_plots',
    'create_classification_analysis',
    'create_prediction_confidence_plot',
    'print_final_sonar_summary'
]