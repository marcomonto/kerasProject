# Moduli per l'analisi breast cancer
# Package contenente tutti i moduli di supporto per l'analisi

from .data_handler import setup_reproducibility, load_and_prepare_data
from .models import create_basic_model, create_funnel_model
from .training import (
    train_and_evaluate,
    run_epochs_experiment, 
    run_batch_size_experiment, 
    run_architecture_experiment
)
from .visualization import create_visualizations, generate_report

__all__ = [
    'setup_reproducibility',
    'load_and_prepare_data',
    'create_basic_model',
    'create_funnel_model',
    'train_and_evaluate',
    'run_epochs_experiment',
    'run_batch_size_experiment', 
    'run_architecture_experiment',
    'create_visualizations',
    'generate_report',
]