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

    # GRAFICO 1: Panoramica generale (3 subplot)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_df['epochs'], epochs_df['precision'],
             'bo-', label='Precision', linewidth=2, markersize=8)
    plt.plot(epochs_df['epochs'], epochs_df['recall'],
             'ro-', label='Recall', linewidth=2, markersize=8)
    plt.plot(epochs_df['epochs'], epochs_df['f1_score'],
             'go-', label='F1-Score', linewidth=2, markersize=8)
    plt.xlabel('Numero di Epochs')
    plt.ylabel('Score')
    plt.title('Effetto del numero di Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.subplot(1, 3, 2)
    plt.semilogx(batch_size_df['batch_size'], batch_size_df['precision'],
                 'bo-', label='Precision', linewidth=2, markersize=8)
    plt.semilogx(batch_size_df['batch_size'], batch_size_df['recall'],
                 'ro-', label='Recall', linewidth=2, markersize=8)
    plt.semilogx(batch_size_df['batch_size'], batch_size_df['f1_score'],
                 'go-', label='F1-Score', linewidth=2, markersize=8)
    plt.xlabel('Batch Size (scala logaritmica)')
    plt.ylabel('Score')
    plt.title('Effetto della dimensione del Batch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.subplot(1, 3, 3)
    models_names = architecture_df['model']
    x_pos = range(len(models_names))

    width = 0.25
    plt.bar([x - width for x in x_pos], architecture_df['precision'],
            width, label='Precision', alpha=0.8, color='blue')
    plt.bar(x_pos, architecture_df['recall'], width,
            label='Recall', alpha=0.8, color='red')
    plt.bar([x + width for x in x_pos], architecture_df['f1_score'],
            width, label='F1-Score', alpha=0.8, color='green')

    plt.xlabel('Architettura')
    plt.ylabel('Score')
    plt.title('Confronto Architetture')
    plt.xticks(x_pos, models_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/1.breast_cancer_overview.png',
                dpi=300, bbox_inches='tight')

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

    # GRAFICO 4: Confronto Architetture dettagliato
    plt.figure(figsize=(10, 6))

    metrics = ['precision', 'recall', 'f1_score', 'loss']
    colors = ['blue', 'red', 'green', 'purple']

    x_pos = range(len(architecture_df))
    width = 0.2

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = plt.bar([x + offset for x in x_pos], architecture_df[metric], width,
                       label=metric.replace('_', '-').title(), alpha=0.8, color=colors[i])

        for bar, value in zip(bars, architecture_df[metric]):
            plt.annotate(f'{value:.3f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                         ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Architettura')
    plt.ylabel('Score')
    plt.title('Confronto Dettagliato Architetture')
    plt.xticks(x_pos, architecture_df['model'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/4.architecture_detailed_comparison.png',
                dpi=300, bbox_inches='tight')

    print("\n✅ Creati 4 grafici dettagliati nella directory results/grafici/:")
    print("  1. breast_cancer_overview.png - Panoramica generale")
    print("  2. epochs_detailed_analysis.png - Analisi dettagliata epochs")
    print("  3. batch_size_detailed_analysis.png - Analisi dettagliata batch size")
    print("  4. architecture_detailed_comparison.png - Confronto architetture")


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

    # GRAFICO 1: Confronto F1-Score tra architetture per ogni configurazione
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    configs = basic_results['config_name'].tolist()
    x_pos = range(len(configs))
    width = 0.35

    plt.bar([x - width/2 for x in x_pos], basic_results['f1_score'], width,
            label='Basic Architecture', alpha=0.8, color='blue')
    plt.bar([x + width/2 for x in x_pos], funnel_results['f1_score'], width,
            label='Funnel Architecture', alpha=0.8, color='red')

    # Annotazioni con valori
    for i, (basic_f1, funnel_f1) in enumerate(zip(basic_results['f1_score'], funnel_results['f1_score'])):
        plt.annotate(f'{basic_f1:.3f}', (i - width/2, basic_f1),
                     ha='center', va='bottom', fontweight='bold')
        plt.annotate(f'{funnel_f1:.3f}', (i + width/2, funnel_f1),
                     ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Configurazione')
    plt.ylabel('F1-Score')
    plt.title('F1-Score: Basic vs Funnel Architecture')
    plt.xticks(x_pos, configs, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)

    # GRAFICO 2: Precision comparison
    plt.subplot(2, 2, 2)
    plt.bar([x - width/2 for x in x_pos], basic_results['precision'], width,
            label='Basic Architecture', alpha=0.8, color='green')
    plt.bar([x + width/2 for x in x_pos], funnel_results['precision'], width,
            label='Funnel Architecture', alpha=0.8, color='orange')

    plt.xlabel('Configurazione')
    plt.ylabel('Precision')
    plt.title('Precision: Basic vs Funnel Architecture')
    plt.xticks(x_pos, configs, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)

    # GRAFICO 3: Recall comparison
    plt.subplot(2, 2, 3)
    plt.bar([x - width/2 for x in x_pos], basic_results['recall'], width,
            label='Basic Architecture', alpha=0.8, color='purple')
    plt.bar([x + width/2 for x in x_pos], funnel_results['recall'], width,
            label='Funnel Architecture', alpha=0.8, color='brown')

    plt.xlabel('Configurazione')
    plt.ylabel('Recall')
    plt.title('Recall: Basic vs Funnel Architecture')
    plt.xticks(x_pos, configs, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)

    # GRAFICO 4: Loss comparison
    plt.subplot(2, 2, 4)
    plt.bar([x - width/2 for x in x_pos], basic_results['loss'], width,
            label='Basic Architecture', alpha=0.8, color='darkblue')
    plt.bar([x + width/2 for x in x_pos], funnel_results['loss'], width,
            label='Funnel Architecture', alpha=0.8, color='darkred')

    plt.xlabel('Configurazione')
    plt.ylabel('Loss')
    plt.title('Loss: Basic vs Funnel Architecture')
    plt.xticks(x_pos, configs, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/5.comprehensive_architecture_comparison.png',
                dpi=300, bbox_inches='tight')

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

def generate_report(epochs_results: List[Dict], batch_size_results: List[Dict],
                    architecture_results: List[Dict]) -> None:
    """
    Genera il report testuale dei risultati

    Args:
        epochs_results: Risultati dell'esperimento epochs
        batch_size_results: Risultati dell'esperimento batch size
        architecture_results: Risultati del confronto architetture
    """
    epochs_df = pd.DataFrame(epochs_results)
    batch_size_df = pd.DataFrame(batch_size_results)
    architecture_df = pd.DataFrame(architecture_results)

    print("\n" + "="*60)
    print("REPORT RISULTATI")
    print("="*60)

    print("\n1. EFFETTO DEGLI EPOCHS:")
    print("-" * 25)
    for result in epochs_results:
        print(
            f"Epochs {result['epochs']:3d}: Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, F1={result['f1_score']:.4f}")

    print(
        f"\nMigliori performance con {epochs_df.loc[epochs_df['f1_score'].idxmax(), 'epochs']} epochs")

    print("\n2. EFFETTO DEL BATCH SIZE:")
    print("-" * 27)
    for result in batch_size_results:
        print(
            f"Batch {result['batch_size']:3d}: Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, F1={result['f1_score']:.4f}")

    print(
        f"\nMigliori performance con batch size {batch_size_df.loc[batch_size_df['f1_score'].idxmax(), 'batch_size']}")

    print("\n3. CONFRONTO ARCHITETTURE:")
    print("-" * 25)
    for result in architecture_results:
        print(f"{result['model']:20s}: Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, F1={result['f1_score']:.4f}")

    if architecture_df.loc[0, 'f1_score'] > architecture_df.loc[1, 'f1_score']:
        print("\nL'architettura base performance meglio dell'architettura a imbuto")
    else:
        print("\nL'architettura a imbuto performa meglio dell'architettura base")
