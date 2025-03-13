"""
--------------------------------------------------------
                ML Junior Practical Test.
         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
--------------------------------------------------------
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_roc_curves(results, output_file="roc_curve.png", output_dir="outputs_balanced_data"):
    """
    Gera um gráfico comparando os AUCs do conjunto de teste para os métodos 'euclidiana' e 'cosseno'.
    """
    ks = sorted(results["euclidean"].keys())
    auc_euclidean = [results["euclidean"][k]["test"]["auc"] for k in ks]
    auc_cosine = [results["cosine"][k]["test"]["auc"] for k in ks]
    
    plt.figure(figsize=(8,6))
    plt.plot(ks, auc_euclidean, marker="o", label="Euclidiana")
    plt.plot(ks, auc_cosine, marker="s", label="Cosseno")
    plt.title("Comparação dos AUCs de Teste (média dos folds)")
    plt.xlabel("Valor de k")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    final_path = os.path.join(output_dir, output_file)
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Curva ROC salva em: {final_path}")

def generate_performance_table(results):
    """
    Cria uma tabela (DataFrame) resumindo as métricas macro (média dos folds) de treino e teste,
    para cada valor de k e cada métrica de distância.
    """
    rows = []
    for metric in results.keys():
        for k, info in results[metric].items():
            rows.append({
                "distance_metric": metric,
                "k": k,
                "train_auc": info["train"]["auc"],
                "train_f1": info["train"]["f1"],
                "train_accuracy": info["train"]["accuracy"],
                "train_precision_macro": info["train"]["precision_macro"],
                "train_recall_macro": info["train"]["recall_macro"],
                "train_top_k": info["train"]["top_k"],
                "test_auc": info["test"]["auc"],
                "test_f1": info["test"]["f1"],
                "test_accuracy": info["test"]["accuracy"],
                "test_precision_macro": info["test"]["precision_macro"],
                "test_recall_macro": info["test"]["recall_macro"],
                "test_top_k": info["test"]["top_k"]
            })
    table = pd.DataFrame(rows)
    return table

def plot_class_metrics(results, metric_name="precision", split="test", distance_metric="euclidean", output_file="class_metrics.png", output_dir="outputs_balanced_data"):
    """
    Plota um gráfico de linha para uma métrica por classe (precision, recall ou f1)
    variando o valor de k, para um split (train/test) e para um tipo de distância.
    """
    ks = sorted(results[distance_metric].keys())
    # Agrega os valores para cada classe
    class_values = {}
    for k in ks:
        k_metrics = results[distance_metric][k][split]["class_metrics"]
        for cls, vals in k_metrics.items():
            if cls not in class_values:
                class_values[cls] = []
            class_values[cls].append(vals[metric_name])
    
    plt.figure(figsize=(10,8))
    for cls, vals in class_values.items():
        plt.plot(ks, vals, marker="o", label=cls)
    
    plt.title(f"{metric_name.capitalize()} por Classe ({split}, {distance_metric})")
    plt.xlabel("Valor de k")
    plt.ylabel(metric_name.capitalize())
    plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    final_path = os.path.join(output_dir, output_file)
    plt.tight_layout()
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Gráfico de {metric_name} por classe ({split}, {distance_metric}) salvo em: {final_path}")
