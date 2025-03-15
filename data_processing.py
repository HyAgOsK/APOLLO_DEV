"""
--------------------------------------------------------

                ML Junior Practical Test.

         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
         
--------------------------------------------------------
"""

# Importação das bibliotecas necessárias
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, Normalizer
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
import os

matplotlib.use('Agg')
def plot_transformation_comparison(data_df, output_file="transformation_comparison.png", output_dir="outputs", seed=0):
    """
    Cria dois subconjuntos dos dados:
      - Um com padronização (StandardScaler)
      - Outro com normalização L2 (Normalizer)
    Aplica t-SNE em cada conjunto e gera um gráfico com 4 subplots:
      Linha 1: Projeção t-SNE "pura" (sem clustering) para cada transformação.
      Linha 2: Projeção t-SNE com overlay de clustering – os centroides (marcados com "X")
              e círculos delimitadores (definidos pelo percentil 90 das distâncias dos pontos ao centróide).
    
    Cada coluna representa uma transformação:
      - Coluna 1: Dados padronizados (StandardScaler)
      - Coluna 2: Dados normalizados L2 (Normalizer)
    
    Parâmetros:
      - data_df: DataFrame original com a coluna 'embedding'
      - output_file: Nome do arquivo para salvar o gráfico final.
      - output_dir: Diretório onde o arquivo será salvo.
      - seed: Semente para reprodutibilidade.
    """
    # Empilha os embeddings originais
    embeddings = np.stack(data_df['embedding'].values)
    
    # Cria as duas transformações
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(embeddings)
    
    normalizer = Normalizer(norm='l2')
    X_normalized = normalizer.fit_transform(embeddings)
    
    # Cria cópias do DataFrame para cada transformação
    df_standardized = data_df.copy()
    df_normalized = data_df.copy()
    
    # Armazena os embeddings transformados (opcional)
    df_standardized['embedding_transformed'] = list(X_standardized)
    df_normalized['embedding_transformed'] = list(X_normalized)
    
    # Aplica t-SNE para cada transformação
    tsne_std = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=5000).fit_transform(X_standardized)
    tsne_norm = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=5000).fit_transform(X_normalized)
    
    df_standardized['tsne_x'] = tsne_std[:, 0]
    df_standardized['tsne_y'] = tsne_std[:, 1]
    df_normalized['tsne_x'] = tsne_norm[:, 0]
    df_normalized['tsne_y'] = tsne_norm[:, 1]
    
    # Cria um gráfico com 2 linhas x 2 colunas
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # --- Linha 1: Gráficos puros ---
    # Coluna 1: Standardized (puro)
    groups_std = df_standardized.groupby('syndrome_id')
    for name, group in groups_std:
        axes[0, 0].scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    axes[0, 0].set_title("t-SNE (Padronizado - Puro)")
    axes[0, 0].set_xlabel("t-SNE 1")
    axes[0, 0].set_ylabel("t-SNE 2")
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True)
    
    # Coluna 2: Normalized (puro)
    groups_norm = df_normalized.groupby('syndrome_id')
    for name, group in groups_norm:
        axes[0, 1].scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    axes[0, 1].set_title("t-SNE (Normalizado L2 - Puro)")
    axes[0, 1].set_xlabel("t-SNE 1")
    axes[0, 1].set_ylabel("t-SNE 2")
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 1].grid(True)
    
    # --- Linha 2: Gráficos com Clusters ---
    # Coluna 1: Standardized com clustering
    # Aplica KMeans usando k igual ao número de classes únicas
    unique_std = df_standardized['syndrome_id'].unique()
    k_std = len(unique_std)
    kmeans_std = KMeans(n_clusters=k_std, random_state=seed)
    clusters_std = kmeans_std.fit_predict(tsne_std)
    df_standardized['cluster'] = clusters_std
    groups_std = df_standardized.groupby('syndrome_id')
    for name, group in groups_std:
        axes[1, 0].scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    # Plota os centroides e círculos para Standardized
    centroids_std = kmeans_std.cluster_centers_
    axes[1, 0].scatter(centroids_std[:, 0], centroids_std[:, 1], marker="X", s=200, c="black", label="Centroides")
    # Para cada cluster, plota um círculo delimitador (percentil 90)
    for i, centroid in enumerate(centroids_std):
        cluster_points = df_standardized[df_standardized['cluster'] == i]
        distances = np.sqrt((cluster_points['tsne_x'] - centroid[0])**2 + (cluster_points['tsne_y'] - centroid[1])**2)
        radius = np.percentile(distances, 90)
        circle = plt.Circle(centroid, radius, color='black', fill=False, linestyle='--', linewidth=2, alpha=0.7)
        axes[1, 0].add_patch(circle)
    axes[1, 0].set_title("t-SNE (Padronizado + Clusters)")
    axes[1, 0].set_xlabel("t-SNE 1")
    axes[1, 0].set_ylabel("t-SNE 2")
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1, 0].grid(True)
    
    # Coluna 2: Normalized com clustering
    unique_norm = df_normalized['syndrome_id'].unique()
    k_norm = len(unique_norm)
    kmeans_norm = KMeans(n_clusters=k_norm, random_state=seed)
    clusters_norm = kmeans_norm.fit_predict(tsne_norm)
    df_normalized['cluster'] = clusters_norm
    groups_norm = df_normalized.groupby('syndrome_id')
    for name, group in groups_norm:
        axes[1, 1].scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    centroids_norm = kmeans_norm.cluster_centers_
    axes[1, 1].scatter(centroids_norm[:, 0], centroids_norm[:, 1], marker="X", s=200, c="black", label="Centroides")
    for i, centroid in enumerate(centroids_norm):
        cluster_points = df_normalized[df_normalized['cluster'] == i]
        distances = np.sqrt((cluster_points['tsne_x'] - centroid[0])**2 + (cluster_points['tsne_y'] - centroid[1])**2)
        radius = np.percentile(distances, 90)
        circle = plt.Circle(centroid, radius, color='black', fill=False, linestyle='--', linewidth=2, alpha=0.7)
        axes[1, 1].add_patch(circle)
    axes[1, 1].set_title("t-SNE (Normalizado L2 + Clusters)")
    axes[1, 1].set_xlabel("t-SNE 1")
    axes[1, 1].set_ylabel("t-SNE 2")
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    final_path = os.path.join(output_dir, output_file)
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Gráfico de comparação de transformações (puro e com clusters) salvo em: {final_path}")




def load_and_preprocess_data(pickle_file):
    """
    Carrega o arquivo pickle e realiza o flatten da estrutura hierárquica.
    
    Estrutura dos dados:
      {
        'syndrome_id': {
            'subject_id': {
                'image_id': [320-dimensional embedding]
            }
        }
      }
    
    Retorna três DataFrames com as colunas:
      ['syndrome_id', 'subject_id', 'image_id', 'embedding']
    
    - df_original: embeddings originais (tipo numpy.array)
    - df_normalizado: embeddings com L2 Normalization (norma = 1)
    - df_padronizado: embeddings com padronização (z-score: média 0, desvio padrão 1)
    
    Obs.: Se um embedding for inválido, é registrado em issues.
    """
    issues = []
    
    # 1) Carrega o arquivo pickle
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        
    # 2) Flatten da estrutura hierárquica
    records = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                if embedding is None or subject_id is None or syndrome_id is None or len(embedding) != 320:
                    records.append({
                        'syndrome_id': syndrome_id,
                        'subject_id': subject_id,
                        'image_id': image_id,
                        'embedding_length': len(embedding) if embedding is not None else None
                    })
                    print(f"[ALERTA] Embedding inválido para a imagem {image_id}. Tamanho: {len(embedding) if embedding is not None else 'None'}")
                else:
                    records.append({
                        'syndrome_id': syndrome_id,
                        'subject_id': subject_id,
                        'image_id': image_id,
                        'embedding': np.array(embedding)  # array numpy (320-dim)
                    })
    
    # 3) Cria o DataFrame original
    df_original = pd.DataFrame(records)
    
    # 4) Cria o DataFrame com L2 Normalization
    df_normalizado = df_original.copy()
    normalizer = Normalizer(norm='l2')
    embeddings = np.stack(df_original['embedding'].values)     # shape (N, 320)
    embeddings_l2 = normalizer.fit_transform(embeddings)
    df_normalizado['embedding'] = list(embeddings_l2)
    
    # 5) Cria o DataFrame com padronização (z-score)
    df_padronizado = df_original.copy()
    scaler = StandardScaler()
    embeddings_std = scaler.fit_transform(embeddings)  # Aplicado sobre os dados originais
    df_padronizado['embedding'] = list(embeddings_std)
    
    return df_original, df_normalizado, df_padronizado, issues

            


# =======================
# Função para gerar gráfico da distribuição de imagens por síndrome
# =======================
def plot_images_distribution(df, output_file="images_per_syndrome.png"):
    """
    Gera e salva um gráfico de barras mostrando o número de imagens por síndrome.
    """
    counts = df.groupby('syndrome_id').size().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue')
    plt.title("Distribuição de Imagens por Síndrome")
    plt.xlabel("Syndrome ID")
    plt.ylabel("Número de Imagens")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Gráfico salvo em {output_file}")
    return counts

# =======================
# Função para criar conjunto balanceado via undersampling
# =======================

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def create_balanced_dataset(df, counts, max_factor=2.0, quantile_threshold=0.7, outlier_quantile=0.95):
    """
    Cria um conjunto 'balanceado' com base na proximidade dos embeddings ao centróide de cada classe.
    Além disso, remove outliers em cada classe (acima de outlier_quantile).
    
    Passos:
      1. Para cada classe, remove outliers (amostras cujo embedding está acima do outlier_quantile de distância do centróide).
      2. Se o número de amostras for menor ou igual ao min_count, mantém todas as amostras.
      3. Caso contrário:
         - Calcula o centróide.
         - Define um limiar (quantile_threshold) para selecionar as amostras mais próximas do centróide.
         - Garante um mínimo de min_count amostras e limita a no máximo max_count (max_factor * min_count).
    """
    min_count = counts.min()
    max_count = int(min_count * max_factor)
    balanced_samples = []
    
    for syndrome, group in df.groupby('syndrome_id'):
        X = np.vstack(group['embedding'].values)
        n = len(group)
        
        # Calcula o centróide
        centroid = X.mean(axis=0)
        # Distâncias de cada embedding ao centróide
        distances = np.linalg.norm(X - centroid, axis=1)
        group = group.copy()
        group['distance'] = distances
        
        # 1) Remove outliers acima do outlier_quantile
        dist_cutoff = np.quantile(distances, outlier_quantile)
        group = group[group['distance'] <= dist_cutoff]
        
        # Recalcula X após remoção dos outliers
        X = np.vstack(group['embedding'].values)
        distances = group['distance'].values
        n = len(group)
        
        # Se após remoção de outliers a classe ficou com <= min_count, mantém todas
        if n <= min_count:
            balanced_samples.append(group.drop(columns='distance', errors='ignore'))
            continue
        
        # 2) Se ainda houver mais que min_count, aplicamos a lógica de quantile_threshold
        threshold = np.quantile(distances, quantile_threshold)
        selected = group[group['distance'] <= threshold]
        
        if len(selected) < min_count:
            # Se selecionou menos que min_count, completa com as mais próximas
            selected = group.sort_values(by='distance').iloc[:min_count]
        elif len(selected) > max_count:
            selected = group.sort_values(by='distance').iloc[:max_count]
        
        balanced_samples.append(selected.drop(columns='distance', errors='ignore'))
    
    balanced_df = pd.concat(balanced_samples, axis=0).reset_index(drop=True)
    balanced_classes = counts[counts == min_count].index.tolist()
    imbalanced_classes = counts[counts > min_count].index.tolist()
    
    print(f"Conjunto balanceado criado: min_count = {min_count}, max_count = {max_count}.")
    return balanced_df, balanced_classes, imbalanced_classes




# =======================
# Função para analisar embeddings e gerar matrizes de similaridade e correlação
# =======================
def analyze_embeddings(df, counts, output_prefix="analysis"):
    """
    Seleciona as 10 síndromes mais frequentes, calcula o embedding médio de cada uma,
    gera e salva um heatmap de similaridade por cosseno (como uma matriz de 'confusão' para similaridade)
    e um heatmap da matriz de correlação (Pearson) entre os embeddings médios.
    
    Retorna:
        top_10_syndromes: Lista das 10 síndromes analisadas.
    """
    top_10_syndromes = counts.index.tolist()
    avg_embeddings = {}
    
    for syndrome in top_10_syndromes:
        syndrome_data = df[df['syndrome_id'] == syndrome]
        emb_matrix = np.vstack(syndrome_data['embedding'].values)
        avg_embeddings[syndrome] = emb_matrix.mean(axis=0)
    
    # Cria uma matriz com os embeddings médios
    matrix = np.array([avg_embeddings[s] for s in top_10_syndromes])
    
    # Matriz de similaridade (cosine similarity)
    cos_sim_matrix = cosine_similarity(matrix)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cos_sim_matrix, cmap='viridis')
    plt.title("Similaridade (Cosine) entre Embeddings Médios (Top 10 Síndromes)")
    plt.xticks(ticks=range(len(top_10_syndromes)), labels=top_10_syndromes, rotation=45)
    plt.yticks(ticks=range(len(top_10_syndromes)), labels=top_10_syndromes)
    plt.colorbar(im)
    plt.tight_layout()
    sim_file = f"{output_prefix}_cosine_similarity.png"
    plt.savefig(sim_file, dpi=300)
    plt.close()
    print(f"Heatmap de similaridade salvo em {sim_file}\n\n")
    
    # Matriz de correlação (Pearson) entre os embeddings médios
    corr_matrix = np.corrcoef(matrix)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Matriz de Correlação (Pearson) dos Embeddings Médios (Top 10 Síndromes)")
    plt.xticks(ticks=range(len(top_10_syndromes)), labels=top_10_syndromes, rotation=45)
    plt.yticks(ticks=range(len(top_10_syndromes)), labels=top_10_syndromes)
    plt.colorbar(im)
    plt.tight_layout()
    corr_file = f"{output_prefix}_correlation.png"
    plt.savefig(corr_file, dpi=300)
    plt.close()
    print(f"Heatmap de correlação salvo em {corr_file}\n\n")
    
    return top_10_syndromes