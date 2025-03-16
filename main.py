"""
--------------------------------------------------------
                ML Junior Practical Test.
         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
--------------------------------------------------------
"""
# Importação das bibliotecas necessárias
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

class Utils:
    @staticmethod
    def print_header(title: str):
        """ Função que printa o header do PIPE LINE """
        header = f"\n{'-' * 70}\n{title}\n{'-' * 70}\n"
        print(header)

    @staticmethod
    def create_dir(directory: str):
        """ Função que cria os diretórios """
        os.makedirs(directory, exist_ok=True)
        return directory

    @staticmethod
    def save_dataset(df: pd.DataFrame, output_dir: str, filename: str, description: str):
        """ Função que salva o dataset """
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        print(f"{description} salvo em {path}\n")

    @staticmethod
    def save_issues(issues, output_dir: str):
        """
        Função que salva e verifica os problemas de integridade além do load_and_preprocess_data 
        que retorna os dados originais, normalizados, padronizados e os problemas de integridade, 
        para essa função, dentro do arquivo data_processing.py
        
        """
        if issues:
            issues_df = pd.DataFrame(issues)
            issues_file = os.path.join(output_dir, "integrity_issues.csv")
            issues_df.to_csv(issues_file, index=False)
            print(f"Relatório de problemas salvo em {issues_file}\n")
        else:
            print("Nenhum problema de integridade encontrado.\n")

    @staticmethod
    def print_distribution(data: pd.DataFrame, label: str, output_dir: str):
        """ Função que printa a distribuição quantitativa dos dados """
        
        print(f"Distribuição de imagens por síndrome ({label}):")
        print(data.groupby('syndrome_id').size(), "\n")
        print(f"Quantidade de síndromes: {data['syndrome_id'].nunique()}\n")
        counts = plot_images_distribution(data, output_file=os.path.join(output_dir, "images_per_syndrome.png"))
        print(f"Contagem de imagens por síndrome ({label}): {counts}\n")
        return counts

def aggregate_knn_results(knn_results):
    """
    Para cada métrica de distância, seleciona o melhor composite score, 
    registrando o valor de k e as métricas de teste correspondentes.
    Retorna um composite score geral (média dos melhores) e um dicionário com os detalhes.
    """
    best_details = {}
    for metric, results in knn_results.items():
        best_composite = -float('inf')
        best_k = None
        best_info = None
        for k, info in results.items():
            comp = composite_score(info["test"]["auc"], info["test"]["f1"], info["test"]["top_k"])
            if comp > best_composite:
                best_composite = comp
                best_k = k
                best_info = info["test"]
        best_details[metric] = {
            "best_composite": best_composite,
            "best_k": best_k,
            "metrics": best_info
        }
    overall = np.mean([d["best_composite"] for d in best_details.values()])
    return overall, best_details

def get_transformation_description(key):
    """
    Mapeia a chave da configuração para uma descrição detalhada.
    """
    mapping = {
        "original": "Dados Originais (Puro)",
        "norm": "Dados Originais - Normalizados (L2)",
        "pad": "Dados Originais - Padronizados (z-score)",
        "balance": "Dados Balanceados (Puro)",
        "mixup": "Dados Aumentados Mixup (Puro)",
        "mixup_original": "Dados Aumentados Mixup (Puro)",
        "mixup_norm": "Dados Aumentados Mixup - Normalizados (L2)",
        "mixup_pad": "Dados Aumentados Mixup - Padronizados (z-score)",
        "mixup_balance": "Dados Aumentados Mixup - Balanceados (Puro)",
        "mixup_norm_balance": "Dados Aumentados Mixup - Balanceados (Normalizados - L2)",
        "mixup_pad_balance": "Dados Aumentados Mixup - Balanceados (Padronizados - z-score)"
    }
    return mapping.get(key, key)


class PipelineStep:
    """Interface para um passo do pipeline."""
    def run(self):
        raise NotImplementedError("Subclasses devem implementar o método run.")

class PreprocessingPipeline(PipelineStep):
    """Carrega e pré-processa os dados originais."""
    def __init__(self, pickle_file: str, seed: int):
        self.pickle_file = pickle_file
        self.seed = seed
        self.output_dir = Utils.create_dir(f"outputs_original_data_seed{seed}")

    def run(self):
        Utils.print_header("PIPELINE 0 - Pré-processamento dos dados originais")
        data_orig, data_norm, df_padronizado, issues = load_and_preprocess_data(self.pickle_file)
        print(f"Dados carregados: {len(data_orig)} registros.\n")
        Utils.save_issues(issues, self.output_dir)
        counts = Utils.print_distribution(data_orig, "dados originais", self.output_dir)
        print("Analisando embeddings (heatmaps, correlação, etc.)...")
        analyze_embeddings(data_orig, counts, output_prefix=os.path.join(self.output_dir, "top10"))
        return {
            "original": data_orig,
            "norm": data_norm,
            "pad": df_padronizado,
            "counts": counts,
            "output_dir": self.output_dir
        }

class TransformationPipeline(PipelineStep):
    """Gera transformações e plots (t-SNE e comparação de transformações)."""
    def __init__(self, data: pd.DataFrame, output_dir: str, file_prefix: str, seed: int):
        self.data = data
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.seed = seed

    def run(self):
        plot_transformation_comparison(
            self.data, 
            output_file=f"{self.file_prefix}_transformation.png", 
            output_dir=self.output_dir, 
            seed=self.seed
        )
        print(f"Executando t-SNE para {self.file_prefix}...\n")
        plot_tsne(self.data, output_file=f"{self.file_prefix}_tsne.png", output_dir=self.output_dir, seed=self.seed)

class KNNPipeline(PipelineStep):
    """Executa a classificação KNN e gera os plots de performance."""
    def __init__(self, data: pd.DataFrame, output_dir: str, label: str, seed: int):
        self.data = data
        self.output_dir = output_dir
        self.label = label
        self.seed = seed

    def run(self):
        print(f"\nExecutando classificação KNN ({self.label})...\n")
        knn_results, stored_preds = run_knn_classification(
            self.data, k_range=range(1, 16), n_folds=10, seed=self.seed, 
            output_dir=self.output_dir, store_predictions=True
        )
        self.evaluate(knn_results)
        plot_macro_auc_k(knn_results, output_file=f"Macro_AUC_vs_K_{self.label}.png", output_dir=self.output_dir)
        
        n_classes = self.data['syndrome_id'].nunique()
        plot_true_roc_curves_both(
            stored_preds, n_classes=n_classes, 
            output_file=f"true_roc_curve_{self.label}.png", output_dir=self.output_dir
        )
        plot_macro_roc_comparison(
            stored_preds, n_classes=n_classes, k_value=10,
            output_file=f"macro_roc_comparison_{self.label}.png", output_dir=self.output_dir
        )
        
        table = generate_performance_table(knn_results)
        print(table)
        Utils.save_dataset(table, self.output_dir, f"knn_performance_table_{self.label}.csv", "Tabela de desempenho")
        
        for metric in ["precision", "recall", "f1"]:
            plot_class_metrics_merged(
                knn_results, metric_name=metric, split="test",
                output_file=f"class_{metric}_test_{self.label}.png", output_dir=self.output_dir
            )
        return knn_results, stored_preds

    def evaluate(self, knn_results):
        for metric in ["euclidean", "cosine"]:
            best_k, best_composite = None, -1
            for k, info in knn_results[metric].items():
                comp = composite_score(info["test"]["auc"], info["test"]["f1"], info["test"]["top_k"])
                print(f"[{metric}] k={k}: AUC {info['test']['auc']:.3f}, F1 {info['test']['f1']:.3f}, Acc {info['test']['accuracy']:.3f}, Composite {comp:.3f}")
                if comp > best_composite:
                    best_composite, best_k = comp, k
            print(f"Melhor k para {metric}: {best_k} (composite: {best_composite:.3f})\n")

class MixupPipeline(PipelineStep):
    """Realiza augmentação Mixup e gera os plots correspondentes."""
    def __init__(self, data: pd.DataFrame, output_dir: str, label: str, seed: int, mixup_kwargs: dict):
        self.data = data
        self.output_dir = output_dir
        self.label = label
        self.seed = seed
        self.mixup_kwargs = mixup_kwargs

    def run(self):
        df_mixup = mixup_augmentation(self.data, **self.mixup_kwargs)
        print(f"Mixup ({self.label}): {len(df_mixup)} amostras geradas.\n")
        plot_syndrome_distribution(df_mixup, title="Distribuição após Mixup", name=self.label, output_dir=self.output_dir)
        plot_tsne(df_mixup, output_file=f"t-SNE_apos_{self.label}.png", output_dir=self.output_dir, seed=self.seed)
        counts = plot_images_distribution(df_mixup, output_file=os.path.join(self.output_dir, f"images_per_syndrome_{self.label}.png"))
        print(f"Contagem de imagens por síndrome ({self.label}): {counts}\n")
        analyze_embeddings(df_mixup, counts, output_prefix=os.path.join(self.output_dir, f"top10_{self.label}"))
        return df_mixup

class BalancedDataPipeline(PipelineStep):
    """Cria e salva os dados balanceados, realizando também transformações e classificação."""
    def __init__(self, pickle_file: str, counts_orig, seed: int):
        self.pickle_file = pickle_file
        self.counts_orig = counts_orig
        self.seed = seed

    def run(self):
        """ Função que executa o pipeline de balanceamento dos dados """
        
        Utils.print_header("PIPELINE 4 - Balanceamento dos Dados e Normalização")
        output_dir = Utils.create_dir(f"outputs_balanced_data_seed{self.seed}")
        data_orig, data_norm, df_padronizado, issues = load_and_preprocess_data(self.pickle_file)
        
        balanced_df, _, _ = create_balanced_dataset(data_orig, self.counts_orig)
        Utils.save_dataset(balanced_df, output_dir, "balanced_dataset.csv", "Conjunto balanceado (Puro)")
        
        balanced_df_pad, _, _ = create_balanced_dataset(df_padronizado, self.counts_orig)
        Utils.save_dataset(balanced_df_pad, output_dir, "balanced_dataset_padronizado.csv", "Conjunto balanceado e Padronizado")
        
        balanced_df_norm, _, _ = create_balanced_dataset(data_norm, self.counts_orig)
        Utils.save_dataset(balanced_df_norm, output_dir, "balanced_dataset_normalizated.csv", "Conjunto balanceado e Normalizado")
        
        print(f"Dados balanceados: {len(balanced_df)} registros.\n")
        Utils.save_issues(issues, output_dir)
        counts_bal = Utils.print_distribution(balanced_df, "balanceados", output_dir)
        analyze_embeddings(balanced_df, counts_bal, output_prefix=os.path.join(output_dir, "top10"))
        
        TransformationPipeline(balanced_df, output_dir, "balanced", self.seed).run()
        KNNPipeline(balanced_df, output_dir, "balance", self.seed).run()
        KNNPipeline(balanced_df_norm, output_dir, "norm", self.seed).run()
        KNNPipeline(df_padronizado, output_dir, "pad", self.seed).run()
        
        return output_dir, balanced_df


class PipelineRunner:
    """ Executa todos os passos do pipeline e exibe os resultados. """
    def __init__(self, pickle_file: str, seed: int = 0):
        self.pickle_file = pickle_file
        self.seed = seed
        self.pipeline_results = {}  # Armazenará o composite score de cada configuração

    def run_all(self):
        """ Função que executa todos os passos do pipeline """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Pré-processamento
        preprocess = PreprocessingPipeline(self.pickle_file, self.seed)
        preproc_results = preprocess.run()
        data_orig = preproc_results["original"]
        data_norm = preproc_results["norm"]
        df_pad = preproc_results["pad"]
        counts_orig = preproc_results["counts"]
        out_dir_orig = preproc_results["output_dir"]
        
        # Pipeline 1: Dados originais (puro/raw)
        Utils.print_header("PIPELINE 1 - Dados Originais (Puro)")
        TransformationPipeline(data_orig, out_dir_orig, "original", self.seed).run()
        knn_orig, _ = KNNPipeline(data_orig, out_dir_orig, "original", self.seed).run()
        self.pipeline_results["original"] = aggregate_knn_results(knn_orig)
        
        # Pipeline 2: Dados normalizados
        Utils.print_header("PIPELINE 2 - Dados Normalizados")
        knn_norm, _ = KNNPipeline(data_norm, out_dir_orig, "norm", self.seed).run()
        self.pipeline_results["norm"] = aggregate_knn_results(knn_norm)
        
        # Pipeline 3: Dados padronizados
        Utils.print_header("PIPELINE 3 - Dados Padronizados")
        knn_pad, _ = KNNPipeline(df_pad, out_dir_orig, "pad", self.seed).run()
        self.pipeline_results["pad"] = aggregate_knn_results(knn_pad)
        print("\n------ Dados originais processados ------\n")
        
        # Pipeline 4: Dados balanceados (puro)
        balanced_pipeline = BalancedDataPipeline(self.pickle_file, counts_orig, self.seed)
        balanced_out_dir, balanced_df = balanced_pipeline.run()
        knn_balance, _ = KNNPipeline(balanced_df, balanced_out_dir, "balance", self.seed).run()
        self.pipeline_results["balance"] = aggregate_knn_results(knn_balance)
        print("\n------ Dados balanceados processados ------\n")
        
        # Pipeline Mixup: Dados aumentados via Mixup (usando dados originais)
        Utils.print_header("PIPELINE 4 - Dados aumentados Mixup (Puro)")
        out_dir_mixup = Utils.create_dir(f"outputs_original_data_mixup_seed{self.seed}")
        # Recarrega dados originais para integridade
        data_orig, data_norm, df_pad, _ = load_and_preprocess_data(self.pickle_file)
        mixup_orig = MixupPipeline(data_orig, out_dir_mixup, "Mixup", self.seed, 
                                   {"n_samples": 1000, "alpha": 0.5, "random_state": self.seed}).run()
        knn_mixup, _ = KNNPipeline(mixup_orig, out_dir_mixup, "mixup", self.seed).run()
        self.pipeline_results["mixup"] = aggregate_knn_results(knn_mixup)
        
        # Augmentações adicionais: SMOTE e Ruído (para visualização)
        df_smote = smote_augmentation(data_orig, random_state=42, k_neighbors=5)
        print(f"SMOTE: {len(df_smote)} amostras.\n")
        plot_syndrome_distribution(df_smote, title="Distribuição após SMOTE", name="SMOTE", output_dir=out_dir_mixup)
        plot_tsne(df_smote, output_file="t-SNE_apos_SMOTE.png", output_dir=out_dir_mixup)
        
        df_noise = noise_augmentation(data_orig, noise_std=0.01, n_new_per_sample=1)
        print(f"Ruído: {len(df_noise)} amostras.\n")
        plot_syndrome_distribution(df_noise, title="Distribuição após Ruído", name="Ruído", output_dir=out_dir_mixup)
        plot_tsne(df_noise, output_file="t-SNE_apos_Ruido.png", output_dir=out_dir_mixup)
        
        df_mixup = mixup_augmentation(data_orig, n_samples=1000, alpha=0.5, random_state=0)
        knn_mixup_orig, _ = KNNPipeline(data_orig, out_dir_mixup, "mixup", self.seed).run()
        TransformationPipeline(df_mixup, out_dir_mixup, "tsne_plot_normL2_vs_pad_df_mixup", self.seed).run()
        self.pipeline_results["mixup_original"] = aggregate_knn_results(knn_mixup_orig)
        print("\n------ Mixup em dados originais processado ------\n")
        
        # Pipeline Mixup em dados normalizados
        Utils.print_header("PIPELINE 5 - Dados aumentados Mixup e Normalizados")
        knn_mixup_norm, _ = KNNPipeline(data_norm, out_dir_mixup, "mixup_norm", self.seed).run()
        self.pipeline_results["mixup_norm"] = aggregate_knn_results(knn_mixup_norm)
        
        # Pipeline Mixup em dados padronizados
        Utils.print_header("PIPELINE 6 - Dados aumentados Mixup e Padronizados")
        knn_mixup_pad, _ = KNNPipeline(df_pad, out_dir_mixup, "mixup_pad", self.seed).run()
        self.pipeline_results["mixup_pad"] = aggregate_knn_results(knn_mixup_pad)
        print("\n------ Mixup em dados norm e pad processado ------\n")
        
        # Pipeline Mixup em dados balanceados (puro)
        Utils.print_header("PIPELINE 8 - Dados aumentados Mixup e Balanceados (Puro)")
        out_dir_bal_mixup = Utils.create_dir(f"outputs_balanced_data_mixup_seed{self.seed}")
        mixup_bal = MixupPipeline(balanced_df, out_dir_bal_mixup, "Mixup", self.seed, 
                                  {"n_samples": 1000, "alpha": 0.5, "random_state": self.seed}).run()
        mixup_bal, _, _ = create_balanced_dataset(mixup_bal, counts_orig)
        Utils.save_dataset(mixup_bal, out_dir_bal_mixup, "balanced_dataset_mixup_balance.csv", "Conjunto balanceado (Mixup)")
        counts = plot_images_distribution(mixup_bal, output_file=os.path.join(out_dir_bal_mixup, "images_per_syndrome_mixup.png"))
        analyze_embeddings(mixup_bal, counts, output_prefix=os.path.join(out_dir_bal_mixup, "top10_mixup"))
        TransformationPipeline(mixup_bal, out_dir_bal_mixup, "tsne_plot_normL2_vs_pad_df_mixup", self.seed).run()
        knn_mixup_bal, _ = KNNPipeline(mixup_bal, out_dir_bal_mixup, "mixup", self.seed).run()
        self.pipeline_results["mixup_balance"] = aggregate_knn_results(knn_mixup_bal)
        
        # Pipeline Mixup em dados balanceados e Normalizados
        Utils.print_header("PIPELINE 9 - Dados aumentados Mixup, Balanceados e Normalizados")
        knn_mixup_norm_bal, _ = KNNPipeline(data_norm, out_dir_bal_mixup, "mixup_norm", self.seed).run()
        self.pipeline_results["mixup_norm_balance"] = aggregate_knn_results(knn_mixup_norm_bal)
        
        # Pipeline Mixup em dados balanceados e Padronizados
        Utils.print_header("PIPELINE 10 - Dados aumentados Mixup, Balanceados e Padronizados")
        knn_mixup_pad_bal, _ = KNNPipeline(df_pad, out_dir_bal_mixup, "mixup_pad", self.seed).run()
        self.pipeline_results["mixup_pad_balance"] = aggregate_knn_results(knn_mixup_pad_bal)
        
        # Exibe as 5 melhores configurações entre todos os pipelines com a descrição das transformações
        top_5 = sorted(self.pipeline_results.items(), key=lambda x: x[1][0], reverse=True)[:5]
        print("\n********** RESULTADO FINAL - TOP 5 CONFIGURAÇÕES **********")
        for i, (config_key, (composite_avg, best_details)) in enumerate(top_5, start=1):
            transform_type = get_transformation_description(config_key)
            print(f"{i}. Configuração: '{config_key}' ({transform_type}) - Composite Score Médio: {composite_avg:.3f}")
            for metric, details in best_details.items():
                print(f"   Métrica de distância: {metric}")
                print(f"      Melhor k: {details['best_k']}")
                print(f"      Composite Score: {details['best_composite']:.3f}")
        print("**************************************************************\n")
        
        return self.pipeline_results


def main(args):
    """ Função principal que executa todo código """
    runner = PipelineRunner(args.pickle_file, seed=0)
    runner.run_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline do ML Junior Practical Test")
    parser.add_argument("--pickle_file", type=str, required=True,
                        help="Caminho para o arquivo pickle com os dados (ex: mini_gm_public_v0.1.p)")
    args = parser.parse_args()
    main(args)
