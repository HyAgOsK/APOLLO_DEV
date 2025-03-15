import os
import argparse
from data_processing import (load_and_preprocess_data, plot_images_distribution, 
                             create_balanced_dataset, analyze_embeddings, plot_transformation_comparison)
from visualization import plot_tsne
from classification import run_knn_classification
from metrics_utils import (plot_macro_auc_k, generate_performance_table, plot_class_metrics_merged, plot_true_roc_curves_both, plot_macro_roc_comparison, composite_score)
import random
import numpy as np
import pandas as pd
from multiple_testing_augmentation_dataset import smote_augmentation, noise_augmentation, mixup_augmentation, plot_syndrome_distribution

# Garantindo a reprodutibilidade
seed = 0
random.seed(seed)
np.random.seed(seed)

def main(args):
    
    pipe0 = """
          --------------------------------------------------------
                PIPELINE 0 - Pre processamento dos dados 
          --------------------------------------------------------
          """
    print(pipe0)
    # PIPELINE 1 - Dados Originais
    outputs_original_data = f"outputs_original_data_seed{seed}"
    os.makedirs(outputs_original_data, exist_ok=True)
    
    print("Carregando e processando os dados...\n\n")
    data_df_original, data_df_norm, df_padronizado, issues = load_and_preprocess_data(args.pickle_file)
    print(f"Dados carregados: {len(data_df_original)} registros.\n\n")
    
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_file = os.path.join(outputs_original_data, "integrity_issues.csv")
        issues_df.to_csv(issues_file, index=False)
        print(f"Relatório de problemas salvo em {issues_file}\n")
    else:
        print("Nenhum problema de integridade encontrado.\n")
        
    print("Realizando análise exploratória nos dados originais...\n\n")
    print("Distribuição de imagens por síndrome:")
    print(data_df_original.groupby('syndrome_id').size(), "\n\n")
    
    counts_orig = plot_images_distribution(data_df_original, output_file=os.path.join(outputs_original_data, "images_per_syndrome.png"))
    print("Contagem de imagens por síndrome (dados originais):")
    print(counts_orig, "\n")
    
    print("Analisando embeddings (heatmaps, correlação, etc.)\n")
    _ = analyze_embeddings(data_df_original, counts_orig, output_prefix=os.path.join(outputs_original_data, "top10"))   
       
    
    pipe1 = """
          --------------------------------------------------------
                    PIPELINE 1 - Dados Originais e Normalizados
          --------------------------------------------------------
          """
    print(pipe1)
    # PIPELINE 1 - Dados Originais
    output_dir_orig = f"outputs_original_data_seed{seed}"
    os.makedirs(output_dir_orig, exist_ok=True)
        
    plot_transformation_comparison(data_df_original, output_file="tsne_plot_normL2_vs_pad.png", output_dir=output_dir_orig, seed=seed)
      
    
    print("Executando t-SNE para dados originais...\n")
    plot_tsne(data_df_original, output_file="tsne_plot_original_data.png", output_dir=output_dir_orig, seed=seed)
    
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados originais)...\n")
    # Armazenando predições para plotagem ROC real (store_predictions=True)
    knn_results_orig, stored_preds_orig = run_knn_classification(data_df_original, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_orig, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados originais
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados originais - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig, output_file="Macro_AUC_vs_K_original.png", output_dir=output_dir_orig)
    
    print("Gerando curvas ROC reais (dados originais) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig, n_classes=len(data_df_original['syndrome_id'].unique()), 
                         output_file="true_roc_curve_original.png", output_dir=output_dir_orig)
    
    plot_macro_roc_comparison(stored_preds_orig, n_classes=len(data_df_original['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_original.png", output_dir=output_dir_orig)
    
    print("Gerando tabela de métricas de desempenho (dados originais)...\n")
    table_orig = generate_performance_table(knn_results_orig)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_orig, "knn_performance_table_original.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados originais
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_original.png"
                plot_class_metrics_merged(knn_results_orig, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_orig)
                
                
                
    print("\n--------------Pipeline concluído com sucesso para os dados originais.---------------\n\n")
    
    
    
    pipe1 = """
          --------------------------------------------------------
                    PIPELINE 2 - Dados Originais e Normalizados
          --------------------------------------------------------
          """
    print(pipe1)
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados normalizados)...\n")
    # Armazenando predições para plotagem ROC real (store_predictions=True)
    knn_results_orig, stored_preds_orig = run_knn_classification(data_df_norm, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_orig, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados originais
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados originais - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig, output_file="Macro_AUC_vs_K_norm.png", output_dir=output_dir_orig)
    
    print("Gerando curvas ROC reais (dados originais) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig, n_classes=len(data_df_norm['syndrome_id'].unique()), 
                         output_file="true_roc_curve_norm.png", output_dir=output_dir_orig)
    
    plot_macro_roc_comparison(stored_preds_orig, n_classes=len(data_df_norm['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_norm.png", output_dir=output_dir_orig)
    
    print("Gerando tabela de métricas de desempenho (dados originais)...\n")
    table_orig = generate_performance_table(knn_results_orig)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_orig, "knn_performance_table_norm.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados originais
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_norm.png"
                plot_class_metrics_merged(knn_results_orig, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_orig)
                  
    
    print("\n--------------Pipeline concluído com sucesso para os dados normalizados.---------------\n\n")
    
    
    
    pipe2 = """
          --------------------------------------------------------
                    PIPELINE 3 - Dados Originais Padronizados
          --------------------------------------------------------
          """
    print(pipe2)
    
    
    knn_results_orig_pad, stored_preds_orig_pad = run_knn_classification(df_padronizado, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_orig, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados originais
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig_pad[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados originais - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig_pad, output_file="Macro_AUC_vs_K_pad.png", output_dir=output_dir_orig)
    
    print("Gerando curvas ROC reais (dados originais) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig_pad, n_classes=len(df_padronizado['syndrome_id'].unique()), 
                         output_file="true_roc_curve_pad.png", output_dir=output_dir_orig)
    
    plot_macro_roc_comparison(stored_preds_orig_pad, n_classes=len(df_padronizado['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_pad.png", output_dir=output_dir_orig)
    
    print("Gerando tabela de métricas de desempenho (dados originais)...\n")
    table_orig = generate_performance_table(knn_results_orig_pad)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_orig, "knn_performance_table.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados originais
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_pad.png"
                plot_class_metrics_merged(knn_results_orig_pad, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_orig)
    
    
    
    
    
    
        print("\n--------------Pipeline concluído com sucesso para os dados originais normalizados e padronizados.---------------\n\n")
    
    
    
    
    
    
    # PIPELINE 2 - Dados Balanceados
    output_dir_bal = f"outputs_balanced_data_seed{seed}"
    os.makedirs(output_dir_bal, exist_ok=True)
    
    pipe3 = """
          ------------------------------------------------------------------
                    PIPELINE 4 - Balanceamento dos Dados e normalização 
          ------------------------------------------------------------------
          """
    print(pipe3)
    
    data_df_original_bal, data_df_norm_bal, df_padronizado, issues = load_and_preprocess_data(args.pickle_file)
    
    
    
    
    balanced_df, _, _ = create_balanced_dataset(data_df_original_bal, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset.csv")
    balanced_df.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado original salvo em {balanced_file}\n")
    
    balanced_df_pad, _, _ = create_balanced_dataset(df_padronizado, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset_padronizado.csv")
    balanced_df_pad.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado e padronizado salvo em {balanced_file}\n")
    
    balanced_df_norm, _, _ = create_balanced_dataset(data_df_norm_bal, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset_normalizated.csv")
    balanced_df_norm.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado e normalizado salvo em {balanced_file}\n")
    
    
    print(f"Dados balanceados carregados: {len(balanced_df)} registros.\n\n")
    
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_file = os.path.join(output_dir_bal, "integrity_issues.csv")
        issues_df.to_csv(issues_file, index=False)
        print(f"Relatório de problemas salvo em {issues_file}\n")
    else:
        print("Nenhum problema de integridade encontrado nos dados balanceados.\n")
    
    
    print("Realizando análise exploratória dos dados balanceados...\n")
    print("Distribuição de imagens por síndrome (balanceados):")
    print(balanced_df.groupby('syndrome_id').size(), "\n\n")
    
    counts_bal = plot_images_distribution(balanced_df, output_file=os.path.join(output_dir_bal, "images_per_syndrome.png"))
    print("Contagem de imagens por síndrome (dados balanceados):")
    print(counts_bal, "\n")
    
    print("Analisando embeddings (dados balanceados)...\n")
    _ = analyze_embeddings(balanced_df, counts_bal, output_prefix=os.path.join(output_dir_bal, "top10"))
    
    
    plot_transformation_comparison(balanced_df, output_file="tsne_plot_balanced_normL2_vs_pad.png", output_dir=output_dir_bal, seed=seed)

    
    
    print("Executando t-SNE para dados balanceados...\n")
    plot_tsne(balanced_df, output_file="tsne_plot_original_data_balance.png", output_dir=output_dir_bal, seed=seed)
    
    
    print("Executando classificação KNN com validação cruzada (dados balanceados)...\n")
    knn_results_bal, stored_preds_bal = run_knn_classification(balanced_df, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_bal, store_predictions=True)
    
    # Para cada método (euclidiana e cosseno) nos dados balanceados
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_bal[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")
    
    print("Gerando curvas Macro AUC vs K (dados balanceados - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_bal, output_file="Macro_AUC_vs_K_balance.png", output_dir=output_dir_bal)
    
    print("Gerando curvas ROC (dados balanceados) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_bal, n_classes=len(balanced_df['syndrome_id'].unique()), 
                         output_file="true_roc_curve_balance.png", output_dir=output_dir_bal)
    
    plot_macro_roc_comparison(stored_preds_bal, n_classes=len(balanced_df['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_balance.png", output_dir=output_dir_bal)
    
    print("Gerando tabela de métricas de desempenho (dados balanceados)...\n")
    table_bal = generate_performance_table(knn_results_bal)
    print(table_bal)
    table_file_bal = os.path.join(output_dir_bal, "knn_performance_table_balance.csv")
    table_bal.to_csv(table_file_bal, index=False)
    print(f"Tabela de desempenho (balanceados) salva em {table_file_bal}\n")
    
    # Gera gráficos por classe para dados balanceados
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_balance.png"
                plot_class_metrics_merged(knn_results_bal, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_bal)
    
    print("----------------------Pipeline concluído com sucesso para os dados balanceados.------------------\n")
    
    
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados balanceados e normalizados)...\n")
    knn_results_bal, stored_preds_bal = run_knn_classification(balanced_df_norm, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_bal, store_predictions=True)
    
    # Para cada método (euclidiana e cosseno) nos dados balanceados
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_bal[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")
    
    print("Gerando curvas Macro AUC vs K (dados balanceados - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_bal, output_file="Macro_AUC_vs_K_norm.png", output_dir=output_dir_bal)
    
    print("Gerando curvas ROC (dados balanceados) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_bal, n_classes=len(balanced_df_norm['syndrome_id'].unique()), 
                         output_file="true_roc_curve_norm.png", output_dir=output_dir_bal)
    
    plot_macro_roc_comparison(stored_preds_bal, n_classes=len(balanced_df_norm['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_norm.png", output_dir=output_dir_bal)
    
    print("Gerando tabela de métricas de desempenho (dados balanceados)...\n")
    table_bal = generate_performance_table(knn_results_bal)
    print(table_bal)
    table_file_bal = os.path.join(output_dir_bal, "knn_performance_table_norm.csv")
    table_bal.to_csv(table_file_bal, index=False)
    print(f"Tabela de desempenho (balanceados) salva em {table_file_bal}\n")
    
    # Gera gráficos por classe para dados balanceados
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_norm.png"
                plot_class_metrics_merged(knn_results_bal, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_bal)
    
    print("----------------------Pipeline concluído com sucesso para os dados normalizados.------------------\n")
    
    
    
    
    pipe4 = """
          ------------------------------------------------------------------
                    PIPELINE 5 - Balanceamento dos Dados e Padronização 
          ------------------------------------------------------------------
          """
    print(pipe4)
    
    
    print("Executando classificação KNN com validação cruzada (dados padronizados)...\n")
    knn_results_bal_pad, stored_preds_bal_pad = run_knn_classification(df_padronizado, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_bal, store_predictions=True)
    
    # Para cada método (euclidiana e cosseno) nos dados balanceados
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_bal_pad[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")
    
    print("Gerando curvas Macro AUC vs K (dados balanceados e padronizados  - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_bal_pad, output_file="Macro_AUC_vs_K_pad.png", output_dir=output_dir_bal)
    
    print("Gerando curvas ROC (dados balanceados) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_bal_pad, n_classes=len(balanced_df_norm['syndrome_id'].unique()), 
                         output_file="true_roc_curve_pad.png", output_dir=output_dir_bal)
    
    plot_macro_roc_comparison(stored_preds_bal_pad, n_classes=len(balanced_df_norm['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_pad.png", output_dir=output_dir_bal)
    
    print("Gerando tabela de métricas de desempenho (dados balanceados)...\n")
    table_bal = generate_performance_table(knn_results_bal_pad)
    print(table_bal)
    table_file_bal = os.path.join(output_dir_bal, "knn_performance_table_pad.csv")
    table_bal.to_csv(table_file_bal, index=False)
    print(f"Tabela de desempenho (balanceados) salva em {table_file_bal}\n")
    
    # Gera gráficos por classe para dados balanceados
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_pad.png"
                plot_class_metrics_merged(knn_results_bal_pad, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_bal)
    
   
    
    
    print("----------------------Pipeline concluído com sucesso para os dados balanceados e padronizados.------------------\n")
    
    
    
    
    pipe5 = """"
          -----------------------------------------------------------------------------------
                    PIPELINE 6 - preprocessamento - Dados MIXUP augmentation original data
          -----------------------------------------------------------------------------------
    
            """
    
    print(pipe5)
    
    output_dir_orig_mixup = f"outputs_original_data_mixup_seed{seed}"
    os.makedirs(output_dir_orig_mixup, exist_ok=True)
    
    
    print("Carregando e processando os dados...\n\n")
    data_df_original, data_df_norm, df_padronizado, issues = load_and_preprocess_data(args.pickle_file)
    
        # Augmentation mixup
    df_mixup = mixup_augmentation(data_df_original, n_samples=1000, alpha=0.5, random_state=seed)
    print(f"Mixup: {len(df_mixup)} amostras após gerar embeddings sintéticos.\n")
    plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name='Mixup', output_dir=output_dir_orig_mixup)
    plot_tsne(df_mixup, output_file="t-SNE_apos_Mixup", output_dir=output_dir_orig_mixup)
    
    # Augmentation SMOTE
    df_smote = smote_augmentation(data_df_original, random_state=42, k_neighbors=5)
    print(f"SMOTE: {len(df_smote)} amostras após oversampling.\n")
    plot_syndrome_distribution(df_smote, title="Distribuição após SMOTE", name="SMOTE" ,output_dir=output_dir_orig_mixup)
    plot_tsne(df_smote, output_file="t-SNE_apos_SMOTE", output_dir=output_dir_orig_mixup)

    # Augmentation Ruído
    df_noise = noise_augmentation(data_df_original, noise_std=0.01, n_new_per_sample=1)
    print(f"Ruído: {len(df_noise)} amostras após adicionar perturbação.\n")
    plot_syndrome_distribution(df_noise, title="Distribuição após Ruído", name="Ruído", output_dir=output_dir_orig_mixup)
    plot_tsne(df_noise, output_file="t-SNE_apos_Ruido", output_dir=output_dir_orig_mixup)
    
    print("Realizando análise exploratória nos dados com MIXUP...\n\n")
    print("Distribuição de imagens por síndrome:")
    print(df_mixup.groupby('syndrome_id').size(), "\n\n")
    
    
    print(f"Dados carregados: {len(df_mixup)} registros.\n\n")
    
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_file = os.path.join(output_dir_orig_mixup, "integrity_issues.csv")
        issues_df.to_csv(issues_file, index=False)
        print(f"Relatório de problemas salvo em {issues_file}\n")
    else:
        print("Nenhum problema de integridade encontrado.\n")
        
    print("Realizando análise exploratória nos dados originais...\n\n")
    print("Distribuição de imagens por síndrome:")
    print(df_mixup.groupby('syndrome_id').size(), "\n\n")
    
    counts_orig = plot_images_distribution(df_mixup, output_file=os.path.join(output_dir_orig_mixup, "images_per_syndrome.png"))
    print("Contagem de imagens por síndrome (dados originais):")
    print(counts_orig, "\n")
    
    print("Analisando embeddings (heatmaps, correlação, etc.)\n")
    _ = analyze_embeddings(df_mixup, counts_orig, output_prefix=os.path.join(output_dir_orig_mixup, "top10"))
    
    
    counts_orig = plot_images_distribution(df_mixup, output_file=os.path.join(output_dir_orig_mixup, "images_per_syndrome_mixup.png"))
    print("Contagem de imagens por síndrome (dados originais):")
    print(counts_orig, "\n")
    
    print("Analisando embeddings (heatmaps, correlação, etc.)\n")
    _ = analyze_embeddings(df_mixup, counts_orig, output_prefix=os.path.join(output_dir_orig_mixup, "top10_mixup"))
    
    pipe6 = """"
          ------------------------------------------------------------------------
                    PIPELINE 7 - Dados MIXUP augmentation original data
          ------------------------------------------------------------------------
    
            """
    
    print(pipe6)
    
    plot_transformation_comparison(df_mixup, output_file="tsne_plot_normL2_vs_pad_df_mixup.png", output_dir=output_dir_orig_mixup, seed=seed)
      
    print("Executando t-SNE para dados MIXUP...\n")
    plot_tsne(df_mixup, output_file="tsne_plot_df_mixup.png", output_dir=output_dir_orig_mixup, seed=seed)
    
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados originais com mixup)...\n")
    # Armazenando predições para plotagem ROC real (store_predictions=True)
    knn_results_orig_mixup, stored_preds_orig_mixup = run_knn_classification(df_mixup, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_orig_mixup, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados originais
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig_mixup[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados originais com mixup - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig_mixup, output_file="Macro_AUC_vs_K_mixup.png", output_dir=output_dir_orig_mixup)
    
    print("Gerando curvas ROC reais (dados originais com mixup) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()), 
                         output_file="true_roc_curve_mixup.png", output_dir=output_dir_orig_mixup)
    
    plot_macro_roc_comparison(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_mixup.png", output_dir=output_dir_orig_mixup)
    
    print("Gerando tabela de métricas de desempenho (dados originais)...\n")
    table_orig = generate_performance_table(knn_results_orig_mixup)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_orig_mixup, "knn_performance_table_mixup.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados originais
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_mixup.png"
                plot_class_metrics_merged(knn_results_orig_mixup, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_orig_mixup)
                
                
                
    print("\n--------------Pipeline concluído com sucesso para os dados originais com mixup.---------------\n\n")
    
    
    
    
    
    
    pipe7 = """"
          --------------------------------------------------------
                    PIPELINE 8 - Dados MIXUP augmentation - normalizados
          --------------------------------------------------------
    
            """
    
    print(pipe7)
    
    # Augmentation mixup
    df_mixup = mixup_augmentation(data_df_norm, n_samples=1000, alpha=0.5, random_state=seed)
    print(f"Mixup: {len(df_mixup)} amostras após gerar embeddings sintéticos.\n")
    plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name='Mixup_norm', output_dir=output_dir_orig_mixup)
    plot_tsne(df_mixup, output_file="t-SNE_apos_Mixup_normalizados", output_dir=output_dir_orig_mixup)
    
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados normalizados com mixup)...\n")
    # Armazenando predições para plotagem ROC real (store_predictions=True)
    knn_results_orig_mixup, stored_preds_orig_mixup = run_knn_classification(df_mixup, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_orig_mixup, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados normalizados
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig_mixup[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados normalizados com mixup - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig_mixup, output_file="Macro_AUC_vs_K_mixup.png", output_dir=output_dir_orig_mixup)
    
    print("Gerando curvas ROC reais (dados normalizados com mixup) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()), 
                         output_file="true_roc_curve_mixup_norm.png", output_dir=output_dir_orig_mixup)
    
    plot_macro_roc_comparison(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_mixup_norm.png", output_dir=output_dir_orig_mixup)
    
    print("Gerando tabela de métricas de desempenho (dados normalizados)...\n")
    table_orig = generate_performance_table(knn_results_orig_mixup)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_orig_mixup, "knn_performance_table_mixup_norm.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados normalizados
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados normalizados, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_mixup_norm.png"
                plot_class_metrics_merged(knn_results_orig_mixup, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_orig_mixup)
                
                
                
    print("\n--------------Pipeline concluído com sucesso para os dados normalizados com mixup.---------------\n\n")
    
    
    
    
    pipe8 = """"
          --------------------------------------------------------
                    PIPELINE 8 - Dados MIXUP augmentation padronizados 
          --------------------------------------------------------
    
            """
    
    print(pipe8)
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados padronizados com mixup)...\n")
    
    # Augmentation mixup
    df_mixup = mixup_augmentation(df_padronizado, n_samples=1000, alpha=0.5, random_state=seed)
    print(f"Mixup: {len(df_mixup)} amostras após gerar embeddings sintéticos.\n")
    plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name='Mixup_pad', output_dir=output_dir_orig_mixup)
    plot_tsne(df_mixup, output_file="t-SNE_apos_Mixup_pad", output_dir=output_dir_orig_mixup)
    
    
    # Armazenando predições para plotagem ROC real (store_predictions=True)
    knn_results_orig_mixup, stored_preds_orig_mixup = run_knn_classification(df_mixup, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_orig_mixup, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados originais
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig_mixup[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados padronizados com mixup - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig_mixup, output_file="Macro_AUC_vs_K_mixup_pad.png", output_dir=output_dir_orig_mixup)
    
    print("Gerando curvas ROC reais (dados padronizados com mixup) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()), 
                         output_file="true_roc_curve_mixup_pad.png", output_dir=output_dir_orig_mixup)
    
    plot_macro_roc_comparison(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_mixup_pad.png", output_dir=output_dir_orig_mixup)
    
    print("Gerando tabela de métricas de desempenho (dados padronizados)...\n")
    table_orig = generate_performance_table(knn_results_orig_mixup)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_orig_mixup, "knn_performance_table_mixup_pad.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados originais
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_mixup_pad.png"
                plot_class_metrics_merged(knn_results_orig_mixup, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_orig_mixup)
                
                
                
    print("\n--------------Pipeline concluído com sucesso para os dados padronizados com mixup.---------------\n\n")
    
    
    

    
    pipe9 = """"
          -----------------------------------------------------------------------------------
                    PIPELINE 9 - preprocessamento - Dados MIXUP augmentation balanced data
          -----------------------------------------------------------------------------------
    
            """
    
    print(pipe9)
    
    output_dir_balanced_mixup = f"outputs_balanced_data_mixup_seed{seed}"
    os.makedirs(output_dir_balanced_mixup, exist_ok=True)
    
    
    print("Carregando e processando os dados...\n\n")
    data_df_original, data_df_norm, df_padronizado, issues = load_and_preprocess_data(args.pickle_file)

    
    
        # Augmentation mixup
    df_mixup = mixup_augmentation(balanced_df, n_samples=1000, alpha=0.5, random_state=seed)
    print(f"Mixup: {len(df_mixup)} amostras após gerar embeddings sintéticos.\n")
    plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name='Mixup', output_dir=output_dir_balanced_mixup)
    plot_tsne(df_mixup, output_file="t-SNE_apos_Mixup", output_dir=output_dir_balanced_mixup)
    
    
    df_mixup, _, _ = create_balanced_dataset(df_mixup, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset_mixup_balance.csv")
    df_mixup.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado original salvo em {balanced_file}\n")
    
    print("Realizando análise exploratória nos dados com MIXUP...\n\n")
    print("Distribuição de imagens por síndrome:")
    print(df_mixup.groupby('syndrome_id').size(), "\n\n")
    
    
    print(f"Dados carregados: {len(df_mixup)} registros.\n\n")
    
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_file = os.path.join(output_dir_balanced_mixup, "integrity_issues.csv")
        issues_df.to_csv(issues_file, index=False)
        print(f"Relatório de problemas salvo em {issues_file}\n")
    else:
        print("Nenhum problema de integridade encontrado.\n")
        
    print("Realizando análise exploratória nos dados originais...\n\n")
    print("Distribuição de imagens por síndrome:")
    print(df_mixup.groupby('syndrome_id').size(), "\n\n")
    
    counts_orig = plot_images_distribution(df_mixup, output_file=os.path.join(output_dir_balanced_mixup, "images_per_syndrome.png"))
    print("Contagem de imagens por síndrome (dados originais):")
    print(counts_orig, "\n")
    
    print("Analisando embeddings (heatmaps, correlação, etc.)\n")
    _ = analyze_embeddings(df_mixup, counts_orig, output_prefix=os.path.join(output_dir_balanced_mixup, "top10"))
    
    
    counts_orig = plot_images_distribution(df_mixup, output_file=os.path.join(output_dir_balanced_mixup, "images_per_syndrome_mixup.png"))
    print("Contagem de imagens por síndrome (dados originais):")
    print(counts_orig, "\n")
    
    print("Analisando embeddings (heatmaps, correlação, etc.)\n")
    _ = analyze_embeddings(df_mixup, counts_orig, output_prefix=os.path.join(output_dir_balanced_mixup, "top10_mixup"))
    
    pipe6 = """"
          ------------------------------------------------------------------------
                    PIPELINE 10 - Dados MIXUP augmentation original data
          ------------------------------------------------------------------------
    
            """
    
    print(pipe6)
    
    plot_transformation_comparison(df_mixup, output_file="tsne_plot_normL2_vs_pad_df_mixup.png", output_dir=output_dir_balanced_mixup, seed=seed)
      
    print("Executando t-SNE para dados MIXUP...\n")
    plot_tsne(df_mixup, output_file="tsne_plot_df_mixup.png", output_dir=output_dir_balanced_mixup, seed=seed)
    
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados originais com mixup)...\n")
    # Armazenando predições para plotagem ROC real (store_predictions=True)
    knn_results_orig_mixup, stored_preds_orig_mixup = run_knn_classification(df_mixup, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_balanced_mixup, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados originais
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig_mixup[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados originais com mixup - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig_mixup, output_file="Macro_AUC_vs_K_mixup.png", output_dir=output_dir_balanced_mixup)
    
    print("Gerando curvas ROC reais (dados originais com mixup) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()), 
                         output_file="true_roc_curve_mixup.png", output_dir=output_dir_balanced_mixup)
    
    plot_macro_roc_comparison(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_mixup.png", output_dir=output_dir_balanced_mixup)
    
    print("Gerando tabela de métricas de desempenho (dados originais)...\n")
    table_orig = generate_performance_table(knn_results_orig_mixup)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_balanced_mixup, "knn_performance_table_mixup.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados originais
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_mixup.png"
                plot_class_metrics_merged(knn_results_orig_mixup, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_balanced_mixup)
                
                
                
    print("\n--------------Pipeline concluído com sucesso para os dados originais com mixup.---------------\n\n")
    
    
    
    
    
    
    pipe7 = """"
          --------------------------------------------------------
                    PIPELINE 11 - Dados MIXUP augmentation - normalizados
          --------------------------------------------------------
    
            """
    
    print(pipe7)
    
    # Augmentation mixup
    df_mixup = mixup_augmentation(data_df_norm, n_samples=1000, alpha=0.5, random_state=seed)
    print(f"Mixup: {len(df_mixup)} amostras após gerar embeddings sintéticos.\n")
    plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name='Mixup_norm', output_dir=output_dir_balanced_mixup)
    plot_tsne(df_mixup, output_file="t-SNE_apos_Mixup_normalizados", output_dir=output_dir_balanced_mixup)
    
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados normalizados com mixup)...\n")
    # Armazenando predições para plotagem ROC real (store_predictions=True)
    knn_results_orig_mixup, stored_preds_orig_mixup = run_knn_classification(df_mixup, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_balanced_mixup, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados normalizados
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig_mixup[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados normalizados com mixup - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig_mixup, output_file="Macro_AUC_vs_K_mixup.png", output_dir=output_dir_balanced_mixup)
    
    print("Gerando curvas ROC reais (dados normalizados com mixup) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()), 
                         output_file="true_roc_curve_mixup_norm.png", output_dir=output_dir_balanced_mixup)
    
    plot_macro_roc_comparison(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_mixup_norm.png", output_dir=output_dir_balanced_mixup)
    
    print("Gerando tabela de métricas de desempenho (dados normalizados)...\n")
    table_orig = generate_performance_table(knn_results_orig_mixup)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_balanced_mixup, "knn_performance_table_mixup_norm.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados normalizados
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados normalizados, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_mixup_norm.png"
                plot_class_metrics_merged(knn_results_orig_mixup, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_balanced_mixup)
                
                
                
    print("\n--------------Pipeline concluído com sucesso para os dados normalizados com mixup.---------------\n\n")
    
    
    
    
    pipe8 = """"
          --------------------------------------------------------
                    PIPELINE 12 - Dados MIXUP augmentation padronizados 
          --------------------------------------------------------
    
            """
    
    print(pipe8)
    
    
    print("\n\nExecutando classificação KNN com validação cruzada (dados padronizados com mixup)...\n")
    
    # Augmentation mixup
    df_mixup = mixup_augmentation(df_padronizado, n_samples=1000, alpha=0.5, random_state=seed)
    print(f"Mixup: {len(df_mixup)} amostras após gerar embeddings sintéticos.\n")
    plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name='Mixup_pad', output_dir=output_dir_balanced_mixup)
    plot_tsne(df_mixup, output_file="t-SNE_apos_Mixup_pad", output_dir=output_dir_balanced_mixup)
    
    
    # Armazenando predições para plotagem ROC real (store_predictions=True)
    knn_results_orig_mixup, stored_preds_orig_mixup = run_knn_classification(df_mixup, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir_balanced_mixup, store_predictions=True)

    # Para cada método (euclidiana e cosseno) nos dados originais
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results_orig_mixup[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f}, \n Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f} \n => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")

        
    print("Gerando curvas Macro AUC vs K (dados padronizados com mixup - Macro AUC)...\n")
    plot_macro_auc_k(knn_results_orig_mixup, output_file="Macro_AUC_vs_K_mixup_pad.png", output_dir=output_dir_balanced_mixup)
    
    print("Gerando curvas ROC reais (dados padronizados com mixup) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()), 
                         output_file="true_roc_curve_mixup_pad.png", output_dir=output_dir_balanced_mixup)
    
    plot_macro_roc_comparison(stored_preds_orig_mixup, n_classes=len(df_mixup['syndrome_id'].unique()),
                          k_value=10, output_file="macro_roc_comparison_mixup_pad.png", output_dir=output_dir_balanced_mixup)
    
    print("Gerando tabela de métricas de desempenho (dados padronizados)...\n")
    table_orig = generate_performance_table(knn_results_orig_mixup)
    print(table_orig)
    table_file_orig = os.path.join(output_dir_balanced_mixup, "knn_performance_table_mixup_pad.csv")
    table_orig.to_csv(table_file_orig, index=False)
    print(f"Tabela de desempenho salva em {table_file_orig}\n")
    
    # Gera gráficos por classe para cada métrica para dados originais
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]: # train não é relevante para dados balanceados ou dados originais, pois esta com 100% de acurácia
                out_file = f"class_{metric_name}_{split}_mixup_pad.png"
                plot_class_metrics_merged(knn_results_orig_mixup, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir_balanced_mixup)
                
                
                
    print("\n--------------Pipeline concluído com sucesso para os dados padronizados com mixup.---------------\n\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("\n\n**********Pipeline concluído com sucesso. Todos os resultados foram salvos em diretórios específicos.**********\n\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline do ML Junior Practical Test")
    parser.add_argument("--pickle_file", type=str, required=True,
                        help="Caminho para o arquivo pickle com os dados (mini_gm_public_v0.1.p)")
    args = parser.parse_args()
    main(args)