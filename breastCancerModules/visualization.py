import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Dict


def create_visualizations(
        epochs_results: List[Dict],
        batch_size_results: List[Dict],
        architecture_results: List[Dict]
) -> None:
    """
    Crea grafici dettagliati per visualizzare i risultati degli esperimenti

    Args:
        epochs_results: Risultati dell'esperimento epochs
        batch_size_results: Risultati dell'esperimento batch size
        architecture_results: Risultati del confronto architetture
    """
    output_dir = "results/breasts_grafici"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("CREAZIONE GRAFICI DETTAGLIATI")
    print("="*60)
    print(f"Salvando grafici in: {output_dir}/")

    epochs_df = pd.DataFrame(epochs_results)
    batch_size_df = pd.DataFrame(batch_size_results)
    architecture_df = pd.DataFrame(architecture_results)


    # GRAFICO 2: Analisi dettagliata Epochs
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_df['epochs'], epochs_df['precision'],
             'bo-', linewidth=3, markersize=10)
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    for i, (x, y) in enumerate(zip(epochs_df['epochs'], epochs_df['precision'])):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_df['epochs'], epochs_df['recall'],
             'ro-', linewidth=3, markersize=10)
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    for i, (x, y) in enumerate(zip(epochs_df['epochs'], epochs_df['recall'])):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_df['epochs'], epochs_df['f1_score'],
             'go-', linewidth=3, markersize=10)
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    for i, (x, y) in enumerate(zip(epochs_df['epochs'], epochs_df['f1_score'])):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_df['epochs'], epochs_df['loss'],
             'mo-', linewidth=3, markersize=10)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(epochs_df['epochs'], epochs_df['loss'])):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/2.epochs_detailed_analysis.png',
                dpi=300, bbox_inches='tight')

    # GRAFICO 3: Analisi dettagliata Batch Size
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.semilogx(batch_size_df['batch_size'],
                 batch_size_df['precision'], 'bo-', linewidth=3, markersize=10)
    plt.xlabel('Batch Size')
    plt.ylabel('Precision')
    plt.title('Precision vs Batch Size')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    for i, (x, y) in enumerate(zip(batch_size_df['batch_size'], batch_size_df['precision'])):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.subplot(2, 2, 2)
    plt.semilogx(batch_size_df['batch_size'],
                 batch_size_df['recall'], 'ro-', linewidth=3, markersize=10)
    plt.xlabel('Batch Size')
    plt.ylabel('Recall')
    plt.title('Recall vs Batch Size')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    for i, (x, y) in enumerate(zip(batch_size_df['batch_size'], batch_size_df['recall'])):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.subplot(2, 2, 3)
    plt.semilogx(batch_size_df['batch_size'],
                 batch_size_df['f1_score'], 'go-', linewidth=3, markersize=10)
    plt.xlabel('Batch Size')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Batch Size')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    for i, (x, y) in enumerate(zip(batch_size_df['batch_size'], batch_size_df['f1_score'])):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.subplot(2, 2, 4)
    plt.semilogx(batch_size_df['batch_size'],
                 batch_size_df['loss'], 'mo-', linewidth=3, markersize=10)
    plt.xlabel('Batch Size')
    plt.ylabel('Loss')
    plt.title('Loss vs Batch Size')
    plt.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(batch_size_df['batch_size'], batch_size_df['loss'])):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/3.batch_size_detailed_analysis.png',
                dpi=300, bbox_inches='tight')


    print("\n✅ Creati 2 grafici dettagliati nella directory results/grafici/:")
    print("  1. epochs_detailed_analysis.png - Analisi dettagliata epochs")
    print("  2. batch_size_detailed_analysis.png - Analisi dettagliata batch size")


def create_comprehensive_architecture_visualization(comprehensive_results: List[Dict]) -> None:
    """
    Crea visualizzazioni per il confronto approfondito delle architetture

    Args:
        comprehensive_results: Risultati del confronto approfondito
    """
    output_dir = "results/breasts_grafici"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("CREAZIONE GRAFICI CONFRONTO APPROFONDITO")
    print("="*60)

    df = pd.DataFrame(comprehensive_results)

    # Separa i risultati per architettura
    basic_results = df[df['architecture'] == 'Basic'].copy()
    funnel_results = df[df['architecture'] == 'Funnel'].copy()
    
    # Get unique config names
    configs = df['config_name'].unique().tolist()

    # GRAFICO 2: Heatmap delle performance
    plt.figure(figsize=(12, 8))

    # Prepara dati per heatmap
    metrics = ['f1_score', 'precision', 'recall', 'loss']
    architectures = ['Basic', 'Funnel']

    heatmap_data = []
    for arch in architectures:
        arch_data = df[df['architecture'] == arch]
        row = []
        for config in configs:
            config_data = arch_data[arch_data['config_name'] == config]
            if not config_data.empty:
                row.append([
                    config_data['f1_score'].values[0],
                    config_data['precision'].values[0],
                    config_data['recall'].values[0],
                    config_data['loss'].values[0]
                ])
            else:
                row.append([0, 0, 0, 1])  # default values
        heatmap_data.append(row)

    # Crea subplot per ogni metrica
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # Estrai dati per questa metrica
        metric_data = []
        for arch_idx in range(len(architectures)):
            metric_row = [heatmap_data[arch_idx][config_idx][i]
                          for config_idx in range(len(configs))]
            metric_data.append(metric_row)

        im = axes[i].imshow(metric_data, cmap='RdYlGn' if metric !=
                            'loss' else 'RdYlGn_r', aspect='auto')

        # Imposta etichette
        axes[i].set_xticks(range(len(configs)))
        axes[i].set_xticklabels(configs, rotation=45)
        axes[i].set_yticks(range(len(architectures)))
        axes[i].set_yticklabels(architectures)
        axes[i].set_title(f'{metric.replace("_", "-").title()}')

        # Aggiungi valori alle celle
        for arch_idx in range(len(architectures)):
            for config_idx in range(len(configs)):
                value = metric_data[arch_idx][config_idx]
                axes[i].text(config_idx, arch_idx, f'{value:.3f}',
                             ha='center', va='center', fontweight='bold',
                             color='white' if value < 0.5 else 'black')

        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/6.architecture_performance_heatmap.png',
                dpi=300, bbox_inches='tight')
    print("\n✅ DONE")

