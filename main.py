"""
--------------------------------------------------------

                ML Junior Practical Test.

         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
         
--------------------------------------------------------
"""
import os
import argparse
import random
import numpy as np
import pandas as pd

from data_processing import (
    load_and_preprocess_data, plot_images_distribution, 
    create_balanced_dataset, analyze_embeddings, plot_transformation_comparison
)
from visualization import plot_tsne
from classification import run_knn_classification
from metrics_utils import (
    plot_macro_auc_k, generate_performance_table, 
    plot_class_metrics_merged, plot_true_roc_curves_both, 
    plot_macro_roc_comparison, composite_score
)
from multiple_testing_augmentation_dataset import (
    smote_augmentation, noise_augmentation, mixup_augmentation, 
    plot_syndrome_distribution
)

# -------------------------------
# Funções utilitárias
# -------------------------------

def print_header(title: str):
    header = f"\n{'-' * 70}\n{title}\n{'-' * 70}\n"
    print(header)

def create_dir(directory: str):
    os.makedirs(directory, exist_ok=True)
    return directory

def save_dataset(df: pd.DataFrame, output_dir: str, filename: str, description: str):
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"{description} salvo em {path}\n")

def save_issues(issues, output_dir: str):
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_file = os.path.join(output_dir, "integrity_issues.csv")
        issues_df.to_csv(issues_file, index=False)
        print(f"Relatório de problemas salvo em {issues_file}\n")
    else:
        print("Nenhum problema de integridade encontrado.\n")

def print_distribution(data: pd.DataFrame, label: str, output_dir: str):
    print(f"Distribuição de imagens por síndrome ({label}):")
    print(data.groupby('syndrome_id').size(), "\n")
    print(f"Quantidade de síndromes total: {len(data['syndrome_id'].unique())}\n")
    counts = plot_images_distribution(data, output_file=os.path.join(output_dir, "images_per_syndrome.png"))
    print(f"Contagem de imagens por síndrome ({label}):")
    print(counts, "\n")
    return counts

# -------------------------------
# Pipeline de Classificação KNN (reutilizado em vários blocos)
# -------------------------------

def evaluate_knn_results(knn_results):
    for metric in ["euclidean", "cosine"]:
        best_k = None
        best_composite = -1
        for k, info in knn_results[metric].items():
            test_auc = info["test"]["auc"]
            test_f1 = info["test"]["f1"]
            test_acc = info["test"]["accuracy"]
            test_prec = info["test"]["precision_macro"]
            test_rec = info["test"]["recall_macro"]
            test_topk = info["test"]["top_k"]
            comp = composite_score(test_auc, test_f1, test_acc, test_prec, test_rec, test_topk)
            print(f"[{metric}] k={k} -> AUC: {test_auc:.3f}, F1: {test_f1:.3f}, Acc: {test_acc:.3f},")
            print(f"    Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, TopK: {test_topk:.3f}")
            print(f"    => Composite: {comp:.3f}")
            if comp > best_composite:
                best_composite = comp
                best_k = k
        print(f"Melhor valor de k para {metric}: {best_k} (composite score: {best_composite:.3f})\n")

def run_knn_pipeline(data, output_dir: str, label: str, seed: int):
    print(f"\nExecutando classificação KNN ({label})...\n")
    knn_results, stored_preds = run_knn_classification(
        data, k_range=range(1, 16), n_folds=10, seed=seed, output_dir=output_dir, store_predictions=True
    )
    evaluate_knn_results(knn_results)
    
    print(f"Gerando curva Macro AUC vs K ({label})...\n")
    plot_macro_auc_k(knn_results, output_file=f"Macro_AUC_vs_K_{label}.png", output_dir=output_dir)
    
    n_classes = len(data['syndrome_id'].unique())
    print(f"Gerando curvas ROC reais ({label}) para k=10...\n")
    plot_true_roc_curves_both(
        stored_preds, n_classes=n_classes, output_file=f"true_roc_curve_{label}.png", output_dir=output_dir
    )
    plot_macro_roc_comparison(
        stored_preds, n_classes=n_classes, k_value=10,
        output_file=f"macro_roc_comparison_{label}.png", output_dir=output_dir
    )
    
    print(f"Gerando tabela de desempenho ({label})...\n")
    table = generate_performance_table(knn_results)
    print(table)
    save_dataset(table, output_dir, f"knn_performance_table_{label}.csv", "Tabela de desempenho")
    
    for metric in ["precision", "recall", "f1"]:
        for split in ["test"]:
            out_file = f"class_{metric}_{split}_{label}.png"
            plot_class_metrics_merged(
                knn_results, metric_name=metric, split=split,
                output_file=out_file, output_dir=output_dir
            )
    return knn_results, stored_preds

# -------------------------------
# Pipeline de Augmentação Mixup (reutilizado)
# -------------------------------

def run_mixup_pipeline(data, output_dir: str, label: str, seed: int, mixup_kwargs: dict):
    df_mixup = mixup_augmentation(data, **mixup_kwargs)
    print(f"Mixup ({label}): {len(df_mixup)} amostras geradas.\n")
    plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name=label, output_dir=output_dir)
    plot_tsne(df_mixup, output_file=f"t-SNE_apos_{label}.png", output_dir=output_dir, seed=seed)
    counts = plot_images_distribution(df_mixup, output_file=os.path.join(output_dir, f"images_per_syndrome_{label}.png"))
    print(f"Contagem de imagens por síndrome ({label}):")
    print(counts, "\n")
    analyze_embeddings(df_mixup, counts, output_prefix=os.path.join(output_dir, f"top10_{label}"))
    return df_mixup

# -------------------------------
# Funções para cada Pipeline
# -------------------------------

def pipeline_preprocess_original(pickle_file: str, seed: int):
    print_header("PIPELINE 0 - Pré-processamento dos dados originais")
    output_dir = create_dir(f"outputs_original_data_seed{seed}")
    data_df_original, data_df_norm, df_padronizado, issues = load_and_preprocess_data(pickle_file)
    print(f"Dados carregados: {len(data_df_original)} registros.\n")
    save_issues(issues, output_dir)
    counts = print_distribution(data_df_original, "dados originais", output_dir)
    print("Analisando embeddings (heatmaps, correlação, etc.)...")
    analyze_embeddings(data_df_original, counts, output_prefix=os.path.join(output_dir, "top10"))
    return data_df_original, data_df_norm, df_padronizado, counts, output_dir

def pipeline_transformation(data, output_dir: str, file_prefix: str, seed: int):
    plot_transformation_comparison(data, output_file=f"{file_prefix}_transformation.png", output_dir=output_dir, seed=seed)
    print(f"Executando t-SNE para {file_prefix}...\n")
    plot_tsne(data, output_file=f"{file_prefix}_tsne.png", output_dir=output_dir, seed=seed)

# Pipeline 4: Balanceamento dos dados (modificado para retornar o DataFrame balanceado)
def pipeline_balanced_data(pickle_file: str, counts_orig, seed: int):
    print_header("PIPELINE 4 - Balanceamento dos Dados e Normalização")
    output_dir = create_dir(f"outputs_balanced_data_seed{seed}")
    data_orig, data_norm, df_padronizado, issues = load_and_preprocess_data(pickle_file)
    
    # Cria e salva os datasets balanceados
    balanced_df, _, _ = create_balanced_dataset(data_orig, counts_orig)
    save_dataset(balanced_df, output_dir, "balanced_dataset.csv", "Conjunto balanceado original")
    
    balanced_df_pad, _, _ = create_balanced_dataset(df_padronizado, counts_orig)
    save_dataset(balanced_df_pad, output_dir, "balanced_dataset_padronizado.csv", "Conjunto balanceado e padronizado")
    
    balanced_df_norm, _, _ = create_balanced_dataset(data_norm, counts_orig)
    save_dataset(balanced_df_norm, output_dir, "balanced_dataset_normalizated.csv", "Conjunto balanceado e normalizado")
    
    print(f"Dados balanceados carregados: {len(balanced_df)} registros.\n")
    save_issues(issues, output_dir)
    counts_bal = print_distribution(balanced_df, "balanceados", output_dir)
    analyze_embeddings(balanced_df, counts_bal, output_prefix=os.path.join(output_dir, "top10"))
    
    pipeline_transformation(balanced_df, output_dir, "balanced", seed)
    run_knn_pipeline(balanced_df, output_dir, "balance", seed)
    run_knn_pipeline(balanced_df_norm, output_dir, "norm", seed)
    run_knn_pipeline(df_padronizado, output_dir, "pad", seed)
    
    return output_dir, balanced_df  

def pipeline_mixup_section(data, output_dir: str, mixup_label: str, seed: int):
    # Executa mixup, gera plots e roda KNN
    df_mixup = run_mixup_pipeline(data, output_dir, mixup_label, seed, 
                                  {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    # Para dados adicionais, podemos aplicar outras augmentações (SMOTE, ruído) se necessário:
    return run_knn_pipeline(df_mixup, output_dir, mixup_label.lower(), seed)

# -------------------------------
# Função Principal
# -------------------------------

def main(args):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    # Pipeline 0: Pré-processamento dos dados originais
    data_orig, data_norm, df_padronizado, counts_orig, out_dir_orig = pipeline_preprocess_original(args.pickle_file, seed)
    
    # Pipeline 1: Transformação e t-SNE dos dados originais
    print_header("PIPELINE 1 - Dados Originais e Normalizados")
    pipeline_transformation(data_orig, out_dir_orig, "original", seed)
    run_knn_pipeline(data_orig, out_dir_orig, "original", seed)
    
    # Pipeline 2: Classificação com dados normalizados
    print_header("PIPELINE 2 - Dados Normalizados")
    run_knn_pipeline(data_norm, out_dir_orig, "norm", seed)
    
    # Pipeline 3: Classificação com dados padronizados
    print_header("PIPELINE 3 - Dados Padronizados")
    run_knn_pipeline(df_padronizado, out_dir_orig, "pad", seed)
    
    print("\n------ Pipeline concluído para dados originais, normalizados e padronizados ------\n")
    
    # Pipeline 4: Balanceamento dos dados
    balanced_out_dir = pipeline_balanced_data(args.pickle_file, counts_orig, seed)
    
    print("\n------ Pipeline concluído para dados balanceados ------\n")
    
    # Pipeline 5: Augmentação Mixup (dados originais)
    print_header("PIPELINE 5 - MIXUP augmentation (dados originais)")
    out_dir_mixup = create_dir(f"outputs_original_data_mixup_seed{seed}")
    # Recarrega os dados originais
    data_orig, data_norm, df_padronizado, _ = load_and_preprocess_data(args.pickle_file)
    run_mixup_pipeline(data_orig, out_dir_mixup, "Mixup", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    # Complementa com SMOTE e ruído
    df_smote = smote_augmentation(data_orig, random_state=42, k_neighbors=5)
    print(f"SMOTE: {len(df_smote)} amostras após oversampling.\n")
    plot_syndrome_distribution(df_smote, title="Distribuição após SMOTE", name="SMOTE", output_dir=out_dir_mixup)
    plot_tsne(df_smote, output_file="t-SNE_apos_SMOTE.png", output_dir=out_dir_mixup)
    df_noise = noise_augmentation(data_orig, noise_std=0.01, n_new_per_sample=1)
    print(f"Ruído: {len(df_noise)} amostras após perturbação.\n")
    plot_syndrome_distribution(df_noise, title="Distribuição após Ruído", name="Ruído", output_dir=out_dir_mixup)
    plot_tsne(df_noise, output_file="t-SNE_apos_Ruido.png", output_dir=out_dir_mixup)
    print("Analisando dados MIXUP (original):")
    print(data_orig.groupby('syndrome_id').size(), "\n")
    counts_mixup = plot_images_distribution(data_orig, output_file=os.path.join(out_dir_mixup, "images_per_syndrome_mixup.png"))
    print("Contagem de imagens por síndrome (Mixup - original):")
    print(counts_mixup, "\n")
    analyze_embeddings(data_orig, counts_mixup, output_prefix=os.path.join(out_dir_mixup, "top10_mixup"))
    pipeline_transformation(data_orig, out_dir_mixup, "tsne_plot_normL2_vs_pad_df_mixup", seed)
    run_knn_pipeline(data_orig, out_dir_mixup, "mixup", seed)
    
    print("\n------ Pipeline concluído para dados originais com mixup ------\n")
    
    # Pipeline 6: MIXUP com dados normalizados
    print_header("PIPELINE 6 - MIXUP augmentation (dados normalizados)")
    run_mixup_pipeline(data_norm, out_dir_mixup, "Mixup_norm", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    run_knn_pipeline(data_norm, out_dir_mixup, "mixup_norm", seed)
    
    # Pipeline 7: MIXUP com dados padronizados
    print_header("PIPELINE 7 - MIXUP augmentation (dados padronizados)")
    run_mixup_pipeline(df_padronizado, out_dir_mixup, "Mixup_pad", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    run_knn_pipeline(df_padronizado, out_dir_mixup, "mixup_pad", seed)
    
    print("\n------ Pipeline concluído para dados padronizados com mixup ------\n")
    
    _, balanced_df = pipeline_balanced_data(args.pickle_file, counts_orig, seed)

    # Pipeline 8: MIXUP para dados balanceados
    print_header("PIPELINE 8 - MIXUP augmentation para dados BALANCEADOS")
    out_dir_bal_mixup = create_dir(f"outputs_balanced_data_mixup_seed{seed}")
    # Aqui, usamos o DataFrame balanceado (balanced_df) e não o diretório
    df_mixup_bal = run_mixup_pipeline(
        balanced_df, out_dir_bal_mixup, "Mixup", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed}
    )
    # (Opcional) Rebalanceia o dataset mixup se necessário:
    df_mixup_bal, _, _ = create_balanced_dataset(df_mixup_bal, counts_orig)
    save_dataset(df_mixup_bal, out_dir_bal_mixup, "balanced_dataset_mixup_balance.csv", "Conjunto balanceado (Mixup)")
    print("Analisando dados MIXUP (balanceados):")
    print(df_mixup_bal.groupby('syndrome_id').size(), "\n")
    counts_mixup_bal = plot_images_distribution(df_mixup_bal, output_file=os.path.join(out_dir_bal_mixup, "images_per_syndrome_mixup.png"))
    print("Contagem de imagens por síndrome (dados mixup balanceados):")
    print(counts_mixup_bal, "\n")
    analyze_embeddings(df_mixup_bal, counts_mixup_bal, output_prefix=os.path.join(out_dir_bal_mixup, "top10_mixup"))
    pipeline_transformation(df_mixup_bal, out_dir_bal_mixup, "tsne_plot_normL2_vs_pad_df_mixup", seed)
    run_knn_pipeline(df_mixup_bal, out_dir_bal_mixup, "mixup", seed)

    # Pipeline 9: MIXUP com dados normalizados para dados balanceados
    print_header("PIPELINE 9 - MIXUP (normalizados) para dados BALANCEADOS")
    run_mixup_pipeline(data_norm, out_dir_bal_mixup, "Mixup_norm", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    run_knn_pipeline(data_norm, out_dir_bal_mixup, "mixup_norm", seed)

    # Pipeline 10: MIXUP com dados padronizados para dados balanceados
    print_header("PIPELINE 10 - MIXUP (padronizados) para dados BALANCEADOS")
    run_mixup_pipeline(df_padronizado, out_dir_bal_mixup, "Mixup_pad", seed, {"n_samples": 1000, "alpha": 0.5, "random_state": seed})
    run_knn_pipeline(df_padronizado, out_dir_bal_mixup, "mixup_pad", seed)

    
    print("\n********** Pipeline concluído com sucesso. Todos os resultados foram salvos em diretórios específicos. **********\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline do ML Junior Practical Test")
    parser.add_argument("--pickle_file", type=str, required=True,
                        help="Caminho para o arquivo pickle com os dados (ex: mini_gm_public_v0.1.p)")
    args = parser.parse_args()
    main(args)
