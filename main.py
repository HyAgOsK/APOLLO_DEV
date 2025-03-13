"""
--------------------------------------------------------
                ML Junior Practical Test.
         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
--------------------------------------------------------
"""

import os
import argparse
from data_processing import (load_and_preprocess_data, plot_images_distribution, 
                             create_balanced_dataset, analyze_embeddings, additional_data_analysis)
from visualization import plot_tsne
from classification import run_knn_classification
from metrics_utils import plot_roc_curves, generate_performance_table, plot_class_metrics
import random
import numpy as np
import pandas as pd

# Garantindo a reprodutibilidade
seed = 0
random.seed(seed)
np.random.seed(seed)

def main(args):
    # PIPELINE 1 - Dados Originais
    output_dir_orig = f"outputs_original_data_seed{seed}"
    os.makedirs(output_dir_orig, exist_ok=True)
    
    print("Carregando e processando os dados...\n\n")
    data_df, issues = load_and_preprocess_data(args.pickle_file)
    print(f"Dados carregados: {len(data_df)} registros.\n\n")
    
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_file = os.path.join(output_dir_orig, "integrity_issues.csv")
        issues_df.to_csv(issues_file, index=False)
        print(f"Relatório de problemas salvo em {issues_file}\n")
    else:
        print("Nenhum problema de integridade encontrado.\n")
    
    pipe1 = """
          --------------------------------------------------------
                    PIPELINE 1 - Dados Originais 
          --------------------------------------------------------
          """
    print(pipe1)
    
    print("Realizando análise exploratória...\n\n")
    print("Distribuição de imagens por síndrome:")
    print(data_df.groupby('syndrome_id').size(), "\n\n")
    
    counts_orig = plot_images_distribution(data_df, output_file=os.path.join(output_dir_orig, "images_per_syndrome.png"))
    print("Contagem de imagens por síndrome (dados originais):")
    print(counts_orig, "\n")
    
    print("Analisando embeddings (heatmaps, correlação, etc.)\n")
    _ = analyze_embeddings(data_df, counts_orig, output_prefix=os.path.join(output_dir_orig, "top10"))
    
    print("Realizando análise adicional (PCA, clustering, outliers)...\n")
    _ , _ = additional_data_analysis(data_df, output_dir_orig, seed=seed)
    
    print("Executando t-SNE para dados originais...\n")
    plot_tsne(data_df, output_file="tsne_plot_nonbalanced.png", output_dir=output_dir_orig, seed=seed)
    
    print("Executando classificação KNN com validação cruzada (dados originais)...\n")
    knn_results_orig = run_knn_classification(data_df, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_orig)
    
    print("Gerando curvas ROC (dados originais)...\n")
    plot_roc_curves(knn_results_orig, output_file="roc_curve.png", output_dir=output_dir_orig)
    
    print("Gerando tabela de métricas de desempenho (dados originais)...\n")
    table_orig = generate_performance_table(knn_results_orig)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_orig, "knn_performance_table.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica (exemplo: precision, recall e f1) para dados originais
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["train", "test"]:
            for dist in ["euclidean", "cosine"]:
                out_file = f"class_{metric_name}_{split}_{dist}.png"
                plot_class_metrics(knn_results_orig, metric_name=metric_name, split=split, distance_metric=dist, output_file=out_file, output_dir=output_dir_orig)
    
    print("\nPipeline concluído com sucesso para os dados originais.\n\n")
    
    # PIPELINE 2 - Dados Balanceados
    output_dir_bal = f"outputs_balanced_data_seed{seed}"
    os.makedirs(output_dir_bal, exist_ok=True)
    
    pipe2 = """
          --------------------------------------------------------
                    PIPELINE 2 - Balanceamento dos Dados 
          --------------------------------------------------------
          """
    print(pipe2)
    
    data_df_bal, issues = load_and_preprocess_data(args.pickle_file)
    print(f"Dados carregados: {len(data_df_bal)} registros.\n\n")
    
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_file = os.path.join(output_dir_bal, "integrity_issues.csv")
        issues_df.to_csv(issues_file, index=False)
        print(f"Relatório de problemas salvo em {issues_file}\n")
    else:
        print("Nenhum problema de integridade encontrado.\n")
    
    # Usando os mesmos counts_orig para balanceamento
    balanced_df, _, _ = create_balanced_dataset(data_df_bal, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset.csv")
    balanced_df.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado salvo em {balanced_file}\n")
    
    print("Realizando análise exploratória dos dados balanceados...\n")
    print("Distribuição de imagens por síndrome (balanceados):")
    print(balanced_df.groupby('syndrome_id').size(), "\n\n")
    
    counts_bal = plot_images_distribution(balanced_df, output_file=os.path.join(output_dir_bal, "images_per_syndrome.png"))
    print("Contagem de imagens por síndrome (dados balanceados):")
    print(counts_bal, "\n")
    
    print("Analisando embeddings (dados balanceados)...\n")
    _ = analyze_embeddings(balanced_df, counts_bal, output_prefix=os.path.join(output_dir_bal, "top10"))
    
    print("Realizando análise adicional (PCA, clustering, outliers) dos dados balanceados...\n")
    _ , _ = additional_data_analysis(balanced_df, output_dir_bal, seed=seed)
    
    print("Executando t-SNE para dados balanceados...\n")
    plot_tsne(balanced_df, output_file="tsne_plot_balanced.png", output_dir=output_dir_bal, seed=seed)
    
    print("Executando classificação KNN com validação cruzada (dados balanceados)...\n")
    knn_results_bal = run_knn_classification(balanced_df, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_bal)
    
    print("Gerando curvas ROC (dados balanceados)...\n")
    plot_roc_curves(knn_results_bal, output_file="roc_curve.png", output_dir=output_dir_bal)
    
    print("Gerando tabela de métricas de desempenho (dados balanceados)...\n")
    table_bal = generate_performance_table(knn_results_bal)
    print(table_bal)
    table_file_bal = os.path.join(output_dir_bal, "knn_performance_table.csv")
    table_bal.to_csv(table_file_bal, index=False)
    print(f"Tabela de desempenho (balanceados) salva em {table_file_bal}\n")
    
    # Gera gráficos por classe para dados balanceados
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["train", "test"]:
            for dist in ["euclidean", "cosine"]:
                out_file = f"class_{metric_name}_{split}_{dist}.png"
                plot_class_metrics(knn_results_bal, metric_name=metric_name, split=split, distance_metric=dist, output_file=out_file, output_dir=output_dir_bal)
    
    print("Pipeline concluído com sucesso para os dados balanceados.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline do ML Junior Practical Test")
    parser.add_argument("--pickle_file", type=str, required=True,
                        help="Caminho para o arquivo pickle com os dados (mini_gm_public_v0.1.p)")
    args = parser.parse_args()
    main(args)
