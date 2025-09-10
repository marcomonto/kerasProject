# Moduli per l'analisi cars regression
# Package contenente tutti i moduli per la predizione prezzo auto

from .data_handler import load_and_prepare_cars_data, get_feature_info
from .model import create_cars_regression_model, create_alternative_cars_model
from .training import train_model_with_validation, compare_predictions, analyze_feature_importance
from .visualization import (create_mse_training_plot, create_comprehensive_analysis, 
                           create_feature_importance_plot, print_final_summary)

__all__ = [
    'load_and_prepare_cars_data',
    'get_feature_info', 
    'create_cars_regression_model',
    'create_alternative_cars_model',
    'train_model_with_validation',
    'compare_predictions',
    'analyze_feature_importance',
    'create_mse_training_plot',
    'create_comprehensive_analysis',
    'create_feature_importance_plot',
    'print_final_summary'
]