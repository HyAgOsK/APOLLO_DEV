"""
--------------------------------------------------------
                ML Junior Practical Test.
         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
--------------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, top_k_accuracy_score, 
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, label_binarize
import os

def run_knn_classification(data_df, 
                           k_range=range(1, 16), 
                           n_folds=10, 
                           seed=42, 
                           top_k=10,
                           output_dir='output_dir',
                           save_csv=True):
    """
    Executa a classificação KNN com duas métricas de distância: Euclidiana e Cosseno.
    Realiza validação cruzada (n_folds) e varre k de 1 a 15.
    
    Para cada k e cada métrica, calcula:
      - Métricas de treino: AUC, F1, acurácia, precisão (macro), recall (macro) e top-k,
        além das métricas por classe a partir da matriz de confusão.
      - Métricas de teste: os mesmos.
    
    Retorna um dicionário com a estrutura:
      results = {
         "euclidean": {
            k: {
               "train": { "auc": ..., "f1": ..., "accuracy": ..., "precision_macro": ...,
                          "recall_macro": ..., "top_k": ..., "class_metrics": { ... } },
               "test": { ... }
            },
            ...
         },
         "cosine": { ... }
      }
    """
    # Preparação dos dados
    X = np.stack(data_df["embedding"].values)
    y = data_df["syndrome_id"].values
    
    # Encode dos rótulos para valores numéricos
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    n_classes = len(classes)
    
    # Binariza rótulos para AUC multiclass (One-vs-Rest)
    y_binarized = label_binarize(y_encoded, classes=np.arange(n_classes))
    
    results = {"euclidean": {}, "cosine": {}}
    
    # Configura validação cruzada
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for metric in ["euclidean", "cosine"]:
        for k in k_range:
            # Agregadores para treino
            train_auc_scores = []
            train_f1_scores = []
            train_acc_scores = []
            train_prec_scores = []
            train_rec_scores = []
            train_topk_scores = []
            conf_mat_train = np.zeros((n_classes, n_classes), dtype=int)
            
            # Agregadores para teste
            test_auc_scores = []
            test_f1_scores = []
            test_acc_scores = []
            test_prec_scores = []
            test_rec_scores = []
            test_topk_scores = []
            conf_mat_test = np.zeros((n_classes, n_classes), dtype=int)
            
            for train_idx, test_idx in skf.split(X, y_encoded):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                
                # Cria o classificador KNN
                clf = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm='brute', weights='distance', n_jobs=-1)
                clf.fit(X_train, y_train)
                
                # Métricas para treino
                y_pred_train = clf.predict(X_train)
                probs_train = clf.predict_proba(X_train)
                conf_mat_train += confusion_matrix(y_train, y_pred_train, labels=np.arange(n_classes))
                y_train_bin = label_binarize(y_train, classes=np.arange(n_classes))
                try:
                    train_auc = roc_auc_score(y_train_bin, probs_train, average='macro', multi_class='ovr')
                except Exception:
                    train_auc = np.nan
                train_auc_scores.append(train_auc)
                train_f1_scores.append(f1_score(y_train, y_pred_train, average='macro'))
                train_acc_scores.append(accuracy_score(y_train, y_pred_train))
                train_prec_scores.append(precision_score(y_train, y_pred_train, average='macro', zero_division=0))
                train_rec_scores.append(recall_score(y_train, y_pred_train, average='macro', zero_division=0))
                effective_top_k = min(top_k, n_classes - 1) if n_classes > 1 else 1
                train_topk_scores.append(top_k_accuracy_score(y_train, probs_train, k=effective_top_k))
                
                # Métricas para teste
                y_pred_test = clf.predict(X_test)
                probs_test = clf.predict_proba(X_test)
                conf_mat_test += confusion_matrix(y_test, y_pred_test, labels=np.arange(n_classes))
                y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
                try:
                    test_auc = roc_auc_score(y_test_bin, probs_test, average='macro', multi_class='ovr')
                except Exception:
                    test_auc = np.nan
                test_auc_scores.append(test_auc)
                test_f1_scores.append(f1_score(y_test, y_pred_test, average='macro'))
                test_acc_scores.append(accuracy_score(y_test, y_pred_test))
                test_prec_scores.append(precision_score(y_test, y_pred_test, average='macro', zero_division=0))
                test_rec_scores.append(recall_score(y_test, y_pred_test, average='macro', zero_division=0))
                test_topk_scores.append(top_k_accuracy_score(y_test, probs_test, k=effective_top_k))
            
            # Agregação das métricas (média dos folds)
            train_auc_mean = np.nanmean(train_auc_scores)
            train_f1_mean = np.mean(train_f1_scores)
            train_acc_mean = np.mean(train_acc_scores)
            train_prec_mean = np.mean(train_prec_scores)
            train_rec_mean = np.mean(train_rec_scores)
            train_topk_mean = np.mean(train_topk_scores)
            
            test_auc_mean = np.nanmean(test_auc_scores)
            test_f1_mean = np.mean(test_f1_scores)
            test_acc_mean = np.mean(test_acc_scores)
            test_prec_mean = np.mean(test_prec_scores)
            test_rec_mean = np.mean(test_rec_scores)
            test_topk_mean = np.mean(test_topk_scores)
            
            # Função auxiliar para extrair métricas por classe da matriz de confusão
            def metrics_from_conf_mat(conf_mat):
                class_info = {}
                for i in range(n_classes):
                    tp = conf_mat[i, i]
                    fn = conf_mat[i, :].sum() - tp
                    fp = conf_mat[:, i].sum() - tp
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    support = conf_mat[i, :].sum()
                    class_info[classes[i]] = {"precision": prec, "recall": rec, "f1": f1_c, "support": int(support)}
                return class_info
            
            train_class_metrics = metrics_from_conf_mat(conf_mat_train)
            test_class_metrics = metrics_from_conf_mat(conf_mat_test)
            
            results[metric][k] = {
                "train": {
                    "auc": train_auc_mean,
                    "f1": train_f1_mean,
                    "accuracy": train_acc_mean,
                    "precision_macro": train_prec_mean,
                    "recall_macro": train_rec_mean,
                    "top_k": train_topk_mean,
                    "class_metrics": train_class_metrics
                },
                "test": {
                    "auc": test_auc_mean,
                    "f1": test_f1_mean,
                    "accuracy": test_acc_mean,
                    "precision_macro": test_prec_mean,
                    "recall_macro": test_rec_mean,
                    "top_k": test_topk_mean,
                    "class_metrics": test_class_metrics
                }
            }
    
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        # Macro metrics CSV
        rows_macro = []
        for metric_type in results.keys():
            for k_val, info in results[metric_type].items():
                train_info = info["train"]
                test_info = info["test"]
                rows_macro.append({
                    "distance_metric": metric_type,
                    "k": k_val,
                    "train_auc": train_info["auc"],
                    "train_f1": train_info["f1"],
                    "train_accuracy": train_info["accuracy"],
                    "train_precision_macro": train_info["precision_macro"],
                    "train_recall_macro": train_info["recall_macro"],
                    "train_top_k": train_info["top_k"],
                    "test_auc": test_info["auc"],
                    "test_f1": test_info["f1"],
                    "test_accuracy": test_info["accuracy"],
                    "test_precision_macro": test_info["precision_macro"],
                    "test_recall_macro": test_info["recall_macro"],
                    "test_top_k": test_info["top_k"]
                })
        df_macro = pd.DataFrame(rows_macro)
        df_macro.to_csv(os.path.join(output_dir, "knn_results_macro.csv"), index=False)
        
        # Métricas por classe CSV
        rows_per_class = []
        for metric_type in results.keys():
            for k_val, info in results[metric_type].items():
                for split in ["train", "test"]:
                    for class_name, vals in info[split]["class_metrics"].items():
                        rows_per_class.append({
                            "distance_metric": metric_type,
                            "k": k_val,
                            "split": split,
                            "class": class_name,
                            "precision": vals["precision"],
                            "recall": vals["recall"],
                            "f1": vals["f1"],
                            "support": vals["support"]
                        })
        df_per_class = pd.DataFrame(rows_per_class)
        df_per_class.to_csv(os.path.join(output_dir, "knn_results_per_class.csv"), index=False)
        
        print(f"[INFO] Resultados macro salvos em: {os.path.join(output_dir, 'knn_results_macro.csv')}")
        print(f"[INFO] Resultados por classe salvos em: {os.path.join(output_dir, 'knn_results_per_class.csv')}")
    
    return results
