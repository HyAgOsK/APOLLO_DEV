import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.cm as cm


def plot_macro_auc_k(results, output_file="Macro_AUC_vs_K.png", output_dir="outputs_balanced_data"):
    """
    Gera um gráfico comparando os macro AUC do conjunto de teste para os métodos 'euclidiana' e 'cosseno',
    em função de k.
    """
    ks = sorted(results["euclidean"].keys())
    auc_euclidean = [results["euclidean"][k]["test"]["auc"] for k in ks]
    auc_cosine = [results["cosine"][k]["test"]["auc"] for k in ks]
    
    plt.figure(figsize=(8,6))
    plt.plot(ks, auc_euclidean, marker="o", label="Euclidiana")
    plt.plot(ks, auc_cosine, marker="s", label="Cosseno")
    plt.title("Macro AUC de Teste vs. Valor de k (média dos folds)")
    plt.xlabel("Valor de k")
    plt.ylabel("Macro AUC")
    plt.legend()
    plt.grid(True)
    final_path = os.path.join(output_dir, output_file)
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Gráfico de Macro AUC salvo em: {final_path}")


def plot_macro_roc_comparison(stored_preds, n_classes, k_value=10, output_file="macro_roc_comparison.png", output_dir="outputs_balanced_data"):
    """
    Gera um gráfico único comparando as curvas macro-average ROC (TPR vs. FPR) 
    para os métodos 'euclidiana' e 'cosseno' para um valor de k especificado.
    Inclui a reta de 45° (random classifier) como referência.
    
    Parâmetros:
      - stored_preds: dicionário com predições armazenadas, com keys "euclidean" e "cosine".
      - n_classes: número total de classes.
      - k_value: valor de k a ser utilizado.
      - output_file: nome do arquivo de saída.
      - output_dir: diretório onde o arquivo será salvo.
    """
    metrics = ["euclidean", "cosine"]
    plt.figure(figsize=(8, 6))
    
    for metric in metrics:
        data = stored_preds[metric][k_value]
        y_true = data["y_true"]
        probs = data["probs"]
        # Binariza os rótulos
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        
        # Calcula a curva ROC para cada classe
        fpr_dict = {}
        tpr_dict = {}
        for c in range(n_classes):
            fpr_dict[c], tpr_dict[c], _ = roc_curve(y_true_bin[:, c], probs[:, c])
        # Calcula a macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, lw=2, 
                 label=f"{metric.capitalize()} Macro-average (AUC={roc_auc_macro:.2f})")
    
    # Reta de 45° (random classifier)
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=2, label="Random")
    
    plt.title(f"Macro-average ROC Comparison (k={k_value})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    final_path = os.path.join(output_dir, output_file)
    plt.tight_layout()
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Macro-average ROC Comparison salvo em: {final_path}")



def plot_true_roc_curves_both(stored_preds, n_classes, k_value=10, output_file="true_roc_curve_both.png", output_dir="outputs_balanced_data"):
    """
    Gera um gráfico com dois subplots (lado a lado) contendo as curvas ROC reais (TPR vs. FPR) 
    para cada classe e a curva macro-average, para os métodos 'euclidiana' e 'cosseno', 
    para um valor de k especificado. Em cada subplot, é adicionada a reta de 45° (random classifier).
    
    Parâmetros:
      - stored_preds: dicionário com predições armazenadas, com keys "euclidean" e "cosine".
      - n_classes: número total de classes.
      - k_value: valor de k a ser plotado.
      - output_file: nome do arquivo de saída.
      - output_dir: diretório onde o arquivo será salvo.
    """
    metrics = ["euclidean", "cosine"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, metric in enumerate(metrics):
        data = stored_preds[metric][k_value]
        y_true = data["y_true"]
        probs = data["probs"]
        
        # Binariza os rótulos
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        
        ax = axes[idx]
        
        # Para cada classe, calcula e plota a curva ROC
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}
        for c in range(n_classes):
            fpr_dict[c], tpr_dict[c], _ = roc_curve(y_true_bin[:, c], probs[:, c])
            roc_auc_dict[c] = auc(fpr_dict[c], tpr_dict[c])
            ax.plot(fpr_dict[c], tpr_dict[c], lw=1, label=f"Classe {c} (AUC={roc_auc_dict[c]:.2f})")
        
        # Curva macro-average
        all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color="black", linestyle="--", lw=2, label=f"Macro-average (AUC={roc_auc_macro:.2f})")
        
        # Reta de 45° (random classifier)
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=2, label="Random")
        
        ax.set_title(f"Curva ROC Real ({metric}, k={k_value})", fontsize=14)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True)
    
    plt.tight_layout()
    final_path = os.path.join(output_dir, output_file)
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Curva ROC real (ambos) salva em: {final_path}")


def generate_performance_table(results):
    """
    Cria uma tabela (DataFrame) resumindo as métricas macro de treino e teste para cada valor de k e para cada métrica.
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


# Função para calcular o índice composto
def composite_score(auc, f1, accuracy, precision, recall, top_k):
    # Se uma das métricas for zero (ou muito baixa), a média geométrica também será baixa.
    return (auc * f1 * accuracy * precision * recall * top_k) ** (1/6)
    
    
def plot_class_metrics_merged(results, metric_name="precision", split="test", output_file="class_metrics_merged.png", output_dir="outputs_balanced_data"):
    """
    Plota um gráfico de linha comparando uma métrica por classe (precision, recall ou f1)
    variando o valor de k para ambos os métodos (euclidiana e cosseno) no mesmo gráfico.
    
    Para cada classe, usa a mesma cor:
      - Linha sólida representa os resultados com distância Euclidiana.
      - Linha tracejada representa os resultados com distância Cosseno.
    
    Parâmetros:
      - results: dicionário de resultados gerado por run_knn_classification.
      - metric_name: nome da métrica ("precision", "recall" ou "f1").
      - split: "train" ou "test".
      - output_file: nome do arquivo de saída.
      - output_dir: diretório onde o arquivo será salvo.
    """
    # Assume que os k's são os mesmos para ambas as métricas
    ks = sorted(results["euclidean"].keys())
    # Obtém as classes (supondo que elas sejam as mesmas para qualquer k e split)
    classes = list(results["euclidean"][ks[0]][split]["class_metrics"].keys())
    
    # Define um colormap para as classes, por exemplo, 'tab10'
    cmap = cm.get_cmap('tab10', len(classes))
    
    plt.figure(figsize=(10,8))
    
    for idx, cls in enumerate(classes):
        # Coleta os valores para cada método para a classe 'cls'
        vals_euc = [results["euclidean"][k][split]["class_metrics"][cls][metric_name] for k in ks]
        vals_cos = [results["cosine"][k][split]["class_metrics"][cls][metric_name] for k in ks]
        
        # Plota para Euclidiana: linha sólida, marcadores "o"
        plt.plot(ks, vals_euc, marker="o", linestyle="solid", color=cmap(idx),
                 label=f"{cls} (Euclidiana)")
        # Plota para Cosseno: linha tracejada, marcadores "s"
        plt.plot(ks, vals_cos, marker="s", linestyle="dashed", color=cmap(idx),
                 label=f"{cls} (Cosseno)")
    
    plt.title(f"{metric_name.capitalize()} por Classe ({split}) - Comparação Euclidiana x Cosseno")
    plt.xlabel("Valor de k")
    plt.ylabel(metric_name.capitalize())
    plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    final_path = os.path.join(output_dir, output_file)
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Gráfico de {metric_name} por classe (merged) salvo em: {final_path}")
