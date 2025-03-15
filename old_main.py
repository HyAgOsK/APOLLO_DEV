import os
import argparse
import random
import numpy as np
import pandas as pd

from data_processing import (load_and_preprocess_data, plot_images_distribution, 
                             create_balanced_dataset, analyze_embeddings, plot_transformation_comparison)
from visualization import plot_tsne
from classification import run_knn_classification
from metrics_utils import (plot_macro_auc_k, generate_performance_table, 
                           plot_class_metrics_merged, plot_true_roc_curves_both, 
                           plot_macro_roc_comparison, composite_score)
from multiple_testing_augmentation_dataset import smote_augmentation, noise_augmentation, mixup_augmentation, plot_syndrome_distribution

# -------------------------------
# Helpers para impressão e salvamento
# -------------------------------

def save_issues(issues, output_dir):
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_file = os.path.join(output_dir, "integrity_issues.csv")
        issues_df.to_csv(issues_file, index=False)
        print(f"Relatório de problemas salvo em {issues_file}\n")
    else:
        print("Nenhum problema de integridade encontrado.\n")


def print_distribution(data, label, output_dir):
    print(f"Distribuição de imagens por síndrome ({label}):")
    print(data.groupby('syndrome_id').size(), "\n")
    counts = plot_images_distribution(data, output_file=os.path.join(output_dir, "images_per_syndrome.png"))
    print(f"Contagem de imagens por síndrome ({label}):")
    print(counts, "\n")
    return counts


# -------------------------------
# Função para avaliação dos resultados do KNN
# -------------------------------

def evaluate_knn_results(knn_results):
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec_mean = info["test"]["precision_macro"]
            test_rec_mean = info["test"]["recall_macro"]
            test_topk_mean = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec_mean, test_rec_mean, test_topk_mean)
            print(f"[{metric}] k={k} -> Test AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f},")
            print(f" Precision: {test_prec_mean:.3f}, Recall: {test_rec_mean:.3f}, TopK: {test_topk_mean:.3f}")
            print(f" => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric} (composite score): {best_k} com composite score: {best_composite:.3f}")


# -------------------------------
# Função para executar a pipeline de classificação KNN
# -------------------------------

def run_knn_pipeline(data, output_dir, label, seed):
    print(f"\nExecutando classificação KNN com validação cruzada ({label})...\n")
    knn_results, stored_preds = run_knn_classification(
        data, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir, store_predictions=True
    )
    evaluate_knn_results(knn_results)
    
    print(f"\nGerando curvas Macro AUC vs K ({label})...\n")
    plot_macro_auc_k(knn_results, output_file=f"Macro_AUC_vs_K_{label}.png", output_dir=output_dir)
    
    n_classes = len(data['syndrome_id'].unique())
    print(f"\nGerando curvas ROC reais ({label}) para k=10, métrica euclidiana e cossine...\n")
    plot_true_roc_curves_both(
        stored_preds, n_classes=n_classes, output_file=f"true_roc_curve_{label}.png", output_dir=output_dir
    )
    plot_macro_roc_comparison(
        stored_preds, n_classes=n_classes, k_value=10, output_file=f"macro_roc_comparison_{label}.png", output_dir=output_dir
    )
    
    print(f"\nGerando tabela de métricas de desempenho ({label})...\n")
    table = generate_performance_table(knn_results)
    print(table)
    table_file = os.path.join(output_dir, f"knn_performance_table_{label}.csv")
    table.to_csv(table_file, index=False)
    print(f"Tabela de desempenho salva em {table_file}\n")
    
    for metric_name in ["precision", "recall", "f1"]:
        for split in ["test"]:
            out_file = f"class_{metric_name}_{split}_{label}.png"
            plot_class_metrics_merged(
                knn_results, metric_name=metric_name, split=split, output_file=out_file, output_dir=output_dir
            )
    return knn_results, stored_preds


# -------------------------------
# Funções para os blocos de augmentação mixup
# -------------------------------

def run_mixup_pipeline(data, output_dir, label, seed, mixup_kwargs):
    # Aplica mixup e gera plots de distribuição e t-SNE
    df_mixup = mixup_augmentation(data, **mixup_kwargs)
    print(f"Mixup: {len(df_mixup)} amostras após gerar embeddings sintéticos.\n")
    plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name=label, output_dir=output_dir)
    plot_tsne(df_mixup, output_file=f"t-SNE_apos_{label}", output_dir=output_dir, seed=seed)
    # Se necessário, repetir análise de distribuição/embeddings
    counts = plot_images_distribution(df_mixup, output_file=os.path.join(output_dir, f"images_per_syndrome_{label}.png"))
    print(f"Contagem de imagens por síndrome ({label}):")
    print(counts, "\n")
    analyze_embeddings(df_mixup, counts, output_prefix=os.path.join(output_dir, f"top10_{label}"))
    return df_mixup


# -------------------------------
# Função principal
# -------------------------------

def main(args):
    # Garantindo a reprodutibilidade
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    # PIPELINE 0: Pré-processamento dos dados originais
    pipe0 = """
          --------------------------------------------------------
                PIPELINE 0 - Pré-processamento dos dados 
          --------------------------------------------------------
          """
    print(pipe0)
    outputs_original_data = f"outputs_original_data_seed{seed}"
    os.makedirs(outputs_original_data, exist_ok=True)
    
    print("Carregando e processando os dados...\n")
    data_df_original, data_df_norm, df_padronizado, issues = load_and_preprocess_data(args.pickle_file)
    print(f"Dados carregados: {len(data_df_original)} registros.\n")
    save_issues(issues, outputs_original_data)
    
    print("Analisando dados originais:")
    counts_orig = print_distribution(data_df_original, "dados originais", outputs_original_data)
    print("Analisando embeddings (heatmaps, correlação, etc.)")
    analyze_embeddings(data_df_original, counts_orig, output_prefix=os.path.join(outputs_original_data, "top10"))
    
    # PIPELINE 1: Dados Originais e Normalizados (transformação e t-SNE)
    pipe1 = """
          --------------------------------------------------------
                    PIPELINE 1 - Dados Originais e Normalizados
          --------------------------------------------------------
          """
    print(pipe1)
    # Plot de transformação e t-SNE para dados originais
    plot_transformation_comparison(
        data_df_original, output_file="tsne_plot_normL2_vs_pad.png", output_dir=outputs_original_data, seed=seed
    )
    print("Executando t-SNE para dados originais...\n")
    plot_tsne(
        data_df_original, output_file="tsne_plot_original_data.png", output_dir=outputs_original_data, seed=seed
    )
    # Classificação KNN para dados originais
    run_knn_pipeline(data_df_original, outputs_original_data, "original", seed)
    
    # PIPELINE 2: Dados Normalizados
    pipe2 = """
          --------------------------------------------------------
                    PIPELINE 2 - Dados Normalizados
          --------------------------------------------------------
          """
    print(pipe2)
    run_knn_pipeline(data_df_norm, outputs_original_data, "norm", seed)
    
    # PIPELINE 3: Dados Padronizados
    pipe3 = """
          --------------------------------------------------------
                    PIPELINE 3 - Dados Padronizados
          --------------------------------------------------------
          """
    print(pipe3)
    run_knn_pipeline(df_padronizado, outputs_original_data, "pad", seed)
    print("\n--------------Pipeline concluído com sucesso para os dados originais, normalizados e padronizados.---------------\n")
    
    # PIPELINE 4: Dados Balanceados (criação, análise e normalização)
    output_dir_bal = f"outputs_balanced_data_seed{seed}"
    os.makedirs(output_dir_bal, exist_ok=True)
    pipe4 = """
          ------------------------------------------------------------------
                    PIPELINE 4 - Balanceamento dos Dados e Normalização 
          ------------------------------------------------------------------
          """
    print(pipe4)
    
    # Recarrega dados para balanceamento
    data_df_original_bal, data_df_norm_bal, df_padronizado_bal, issues_bal = load_and_preprocess_data(args.pickle_file)
    
    # Cria datasets balanceados a partir dos dados originais, padronizados e normalizados
    balanced_df, _, _ = create_balanced_dataset(data_df_original_bal, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset.csv")
    balanced_df.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado original salvo em {balanced_file}\n")
    
    balanced_df_pad, _, _ = create_balanced_dataset(df_padronizado_bal, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset_padronizado.csv")
    balanced_df_pad.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado e padronizado salvo em {balanced_file}\n")
    
    balanced_df_norm, _, _ = create_balanced_dataset(data_df_norm_bal, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset_normalizated.csv")
    balanced_df_norm.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado e normalizado salvo em {balanced_file}\n")
    
    print(f"Dados balanceados carregados: {len(balanced_df)} registros.\n")
    save_issues(issues_bal, output_dir_bal)
    
    print("Analisando dados balanceados:")
    counts_bal = print_distribution(balanced_df, "balanceados", output_dir_bal)
    print("Analisando embeddings (dados balanceados)...")
    analyze_embeddings(balanced_df, counts_bal, output_prefix=os.path.join(output_dir_bal, "top10"))
    
    plot_transformation_comparison(
        balanced_df, output_file="tsne_plot_balanced_normL2_vs_pad.png", output_dir=output_dir_bal, seed=seed
    )
    print("Executando t-SNE para dados balanceados...\n")
    plot_tsne(balanced_df, output_file="tsne_plot_original_data_balance.png", output_dir=output_dir_bal, seed=seed)
    
    # Classificação KNN para dados balanceados (original, normalizados e padronizados)
    run_knn_pipeline(balanced_df, output_dir_bal, "balance", seed)
    run_knn_pipeline(balanced_df_norm, output_dir_bal, "norm", seed)
    run_knn_pipeline(df_padronizado_bal, output_dir_bal, "pad", seed)
    print("\n----------------------Pipeline concluído com sucesso para os dados balanceados.------------------\n")
    
    # PIPELINE 5: Augmentação – Dados MIXUP (usando dados originais)
    pipe5 = """
          -----------------------------------------------------------------------------------
                    PIPELINE 5 - Pré-processamento: Dados MIXUP augmentation (original)
          -----------------------------------------------------------------------------------
          """
    print(pipe5)
    output_dir_orig_mixup = f"outputs_original_data_mixup_seed{seed}"
    os.makedirs(output_dir_orig_mixup, exist_ok=True)
    
    print("Carregando e processando os dados...\n")
    data_df_original, data_df_norm, df_padronizado, issues = load_and_preprocess_data(args.pickle_file)
    
    # Aplicando mixup, SMOTE e Ruído
    df_mixup = run_mixup_pipeline(data_df_original, output_dir_orig_mixup, "Mixup", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    print(f"Mixup: {len(df_mixup)} amostras após gerar embeddings sintéticos.\n")
    
    df_smote = smote_augmentation(data_df_original, random_state=42, k_neighbors=5)
    print(f"SMOTE: {len(df_smote)} amostras após oversampling.\n")
    plot_syndrome_distribution(df_smote, title="Distribuição após SMOTE", name="SMOTE", output_dir=output_dir_orig_mixup)
    plot_tsne(df_smote, output_file="t-SNE_apos_SMOTE", output_dir=output_dir_orig_mixup)
    
    df_noise = noise_augmentation(data_df_original, noise_std=0.01, n_new_per_sample=1)
    print(f"Ruído: {len(df_noise)} amostras após adicionar perturbação.\n")
    plot_syndrome_distribution(df_noise, title="Distribuição após Ruído", name="Ruído", output_dir=output_dir_orig_mixup)
    plot_tsne(df_noise, output_file="t-SNE_apos_Ruido", output_dir=output_dir_orig_mixup)
    
    print("Analisando dados MIXUP (original):")
    print(data_df_original.groupby('syndrome_id').size(), "\n")
    counts_mixup = plot_images_distribution(df_mixup, output_file=os.path.join(output_dir_orig_mixup, "images_per_syndrome_mixup.png"))
    print("Contagem de imagens por síndrome (Mixup - original):")
    print(counts_mixup, "\n")
    analyze_embeddings(df_mixup, counts_mixup, output_prefix=os.path.join(output_dir_orig_mixup, "top10_mixup"))
    
    # Transformação e t-SNE para dados mixup originais
    plot_transformation_comparison(
        df_mixup, output_file="tsne_plot_normL2_vs_pad_df_mixup.png", output_dir=output_dir_orig_mixup, seed=seed
    )
    print("Executando t-SNE para dados MIXUP...\n")
    plot_tsne(df_mixup, output_file="tsne_plot_df_mixup.png", output_dir=output_dir_orig_mixup, seed=seed)
    
    # Classificação KNN para dados originais com mixup
    run_knn_pipeline(df_mixup, output_dir_orig_mixup, "mixup", seed)
    print("\n--------------Pipeline concluído com sucesso para os dados originais com mixup.---------------\n")
    
    # PIPELINE 6: MIXUP usando dados normalizados
    pipe6 = """
          ----------------------------------------------------------------------------------------------------------------
                    PIPELINE 6 - Dados MIXUP augmentation (normalizados)
          ----------------------------------------------------------------------------------------------------------------
          """
    print(pipe6)
    df_mixup_norm = run_mixup_pipeline(data_df_norm, output_dir_orig_mixup, "Mixup_norm", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    run_knn_pipeline(df_mixup_norm, output_dir_orig_mixup, "mixup_norm", seed)
    
    # PIPELINE 7: MIXUP usando dados padronizados
    pipe7 = """
          ----------------------------------------------------------------------------------------------------------------
                    PIPELINE 7 - Dados MIXUP augmentation (padronizados)
          ----------------------------------------------------------------------------------------------------------------
          """
    print(pipe7)
    df_mixup_pad = run_mixup_pipeline(df_padronizado, output_dir_orig_mixup, "Mixup_pad", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    run_knn_pipeline(df_mixup_pad, output_dir_orig_mixup, "mixup_pad", seed)
    print("\n--------------Pipeline concluído com sucesso para os dados padronizados com mixup.---------------\n")
    
    # PIPELINE 8: MIXUP para dados balanceados
    pipe8 = """
          -----------------------------------------------------------------------------------
                    PIPELINE 8 - Dados MIXUP augmentation para dados BALANCEADOS
          -----------------------------------------------------------------------------------
          """
    print(pipe8)
    output_dir_balanced_mixup = f"outputs_balanced_data_mixup_seed{seed}"
    os.makedirs(output_dir_balanced_mixup, exist_ok=True)
    
    print("Carregando e processando os dados para balanceamento...\n")
    data_df_original, data_df_norm, df_padronizado, issues = load_and_preprocess_data(args.pickle_file)
    
    df_mixup_bal = run_mixup_pipeline(balanced_df, output_dir_balanced_mixup, "Mixup", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    # Rebalanceia o dataset mixup
    df_mixup_bal, _, _ = create_balanced_dataset(df_mixup_bal, counts_orig)
    balanced_file = os.path.join(output_dir_bal, "balanced_dataset_mixup_balance.csv")
    df_mixup_bal.to_csv(balanced_file, index=False)
    print(f"Conjunto balanceado (Mixup) salvo em {balanced_file}\n")
    
    print("Analisando dados MIXUP (balanceados):")
    print(df_mixup_bal.groupby('syndrome_id').size(), "\n")
    counts_mixup_bal = plot_images_distribution(df_mixup_bal, output_file=os.path.join(output_dir_balanced_mixup, "images_per_syndrome_mixup.png"))
    print("Contagem de imagens por síndrome (dados mixup balanceados):")
    print(counts_mixup_bal, "\n")
    analyze_embeddings(df_mixup_bal, counts_mixup_bal, output_prefix=os.path.join(output_dir_balanced_mixup, "top10_mixup"))
    
    plot_transformation_comparison(
        df_mixup_bal, output_file="tsne_plot_normL2_vs_pad_df_mixup.png", output_dir=output_dir_balanced_mixup, seed=seed
    )
    print("Executando t-SNE para dados MIXUP (balanceados)...\n")
    plot_tsne(df_mixup_bal, output_file="tsne_plot_df_mixup.png", output_dir=output_dir_balanced_mixup, seed=seed)
    run_knn_pipeline(df_mixup_bal, output_dir_balanced_mixup, "mixup", seed)
    print("\n--------------Pipeline concluído com sucesso para os dados balanceados com mixup.---------------\n")
    
    # PIPELINE 9: MIXUP usando dados normalizados para dados balanceados
    pipe9 = """
          ----------------------------------------------------------------------------------------------------------------
                    PIPELINE 9 - Dados MIXUP augmentation (normalizados) para dados BALANCEADOS
          ----------------------------------------------------------------------------------------------------------------
          """
    print(pipe9)
    df_mixup_bal_norm = run_mixup_pipeline(data_df_norm, output_dir_balanced_mixup, "Mixup_norm", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    run_knn_pipeline(df_mixup_bal_norm, output_dir_balanced_mixup, "mixup_norm", seed)
    
    # PIPELINE 10: MIXUP usando dados padronizados para dados balanceados
    pipe10 = """
          ----------------------------------------------------------------------------------------------------------------
                    PIPELINE 10 - Dados MIXUP augmentation (padronizados) para dados BALANCEADOS
          ----------------------------------------------------------------------------------------------------------------
          """
    print(pipe10)
    df_mixup_bal_pad = run_mixup_pipeline(df_padronizado, output_dir_balanced_mixup, "Mixup_pad", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    run_knn_pipeline(df_mixup_bal_pad, output_dir_balanced_mixup, "mixup_pad", seed)
    
    print("\n\n**********Pipeline concluído com sucesso. Todos os resultados foram salvos em diretórios específicos.**********\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline do ML Junior Practical Test")
    parser.add_argument("--pickle_file", type=str, required=True,
                        help="Caminho para o arquivo pickle com os dados (mini_gm_public_v0.1.p)")
    args = parser.parse_args()
    main(args)
