"""
--------------------------------------------------------

                ML Junior Practical Test.

         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
         
--------------------------------------------------------
"""
# Importando das bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def plot_tsne(data_df, output_file='tsne_normal_clustering.png', output_dir='outputs_balanced_data', seed=0):
    """
    Gera uma figura com dois subplots lado a lado a partir dos embeddings:
    
      - Subplot Esquerdo ("Puro"): t-SNE dos embeddings padronizados, 
        colorido de acordo com o 'syndrome_id'.
      
      - Subplot Direito ("Com Clusters"): t-SNE dos embeddings padronizados 
        com overlay de clustering, mostrando os centroides (marcados com "X") 
        e círculos delimitadores (usando o 90º percentil das distâncias dos pontos ao centróide).
    
    A figura é salva no diretório output_dir.
    """
    # Copia o DataFrame para evitar modificações
    data_copy = data_df.copy()
    
    # Empilha os embeddings em uma matriz
    embeddings = np.stack(data_copy['embedding'].values)
    
    # Padroniza os embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    
    # Aplica t-SNE nos embeddings padronizados
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=10000)
    embeddings_2d = tsne.fit_transform(X_scaled)
    
    # Adiciona as coordenadas 2D ao DataFrame
    data_copy['tsne_x'] = embeddings_2d[:, 0]
    data_copy['tsne_y'] = embeddings_2d[:, 1]
    
    # Configura a figura com 1 linha e 2 colunas
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot Esquerdo: Gráfico Puro (sem clustering)
    groups = data_copy.groupby('syndrome_id')
    for name, group in groups:
        axes[0].scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    axes[0].set_title("t-SNE (Puro)")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(title="Syndrome ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True)
    
    # Subplot Direito: Gráfico com Clusters
    # Aplica KMeans com k igual ao número de classes únicas
    unique_classes = data_copy['syndrome_id'].unique()
    k = len(unique_classes)
    kmeans = KMeans(n_clusters=k, random_state=seed)
    clusters = kmeans.fit_predict(embeddings_2d)
    data_copy['cluster'] = clusters
    
    # Plota os pontos, iguais ao gráfico puro
    groups = data_copy.groupby('syndrome_id')
    for name, group in groups:
        axes[1].scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    
    # Plota os centroides
    centroids = kmeans.cluster_centers_
    axes[1].scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, c="black", label="Centroides")
    
    # Para cada cluster, calcula as distâncias dos pontos ao centróide e plota um círculo delimitador
    for i, centroid in enumerate(centroids):
        cluster_points = data_copy[data_copy['cluster'] == i]
        distances = np.sqrt((cluster_points['tsne_x'] - centroid[0])**2 + (cluster_points['tsne_y'] - centroid[1])**2)
        radius = np.percentile(distances, 90)
        circle = plt.Circle(centroid, radius, color='black', fill=False, linestyle='--', linewidth=2, alpha=0.7)
        axes[1].add_patch(circle)
    
    axes[1].set_title("t-SNE + Clusters (Centroides e Delimitação)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(title="Syndrome ID / Centroides", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].grid(True)
    
    plt.tight_layout()
    final_path = os.path.join(output_dir, output_file)
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Figura com ambos os gráficos t-SNE salvo em: {final_path}")
