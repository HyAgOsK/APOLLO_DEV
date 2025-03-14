import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def plot_tsne(data_df, output_file='tsne_plot.png', output_dir='outputs_balanced_data', seed=42):
    """
    Gera dois gráficos t-SNE a partir dos embeddings:
    
    1) Um gráfico "puro", sem centroides nem clusters, apenas colorido por syndrome_id.
    2) Um gráfico "com centroides e clusters", usando KMeans com k = número de classes.
    
    Ambos são salvos no diretório output_dir.
    """
    # Copia o DataFrame para evitar modificações
    data_copy = data_df.copy()
    
    # Empilha os embeddings em uma matriz
    embeddings = np.stack(data_copy['embedding'].values)
    
    # Padroniza os embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    
    # Aplica t-SNE
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=5000)
    embeddings_2d = tsne.fit_transform(X_scaled)
    
    # Adiciona as coordenadas 2D ao DataFrame
    data_copy['tsne_x'] = embeddings_2d[:, 0]
    data_copy['tsne_y'] = embeddings_2d[:, 1]
    
    # =========================
    # 1) Gráfico Puro
    # =========================
    plt.figure(figsize=(12, 10))
    groups = data_copy.groupby('syndrome_id')
    for name, group in groups:
        plt.scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    
    plt.title("t-SNE de Embeddings Padronizados (Puro)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Syndrome ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    pure_path = os.path.join(output_dir, output_file.replace(".png", "_pure.png"))
    plt.savefig(pure_path, dpi=300)
    plt.close()
    print(f"Gráfico t-SNE (puro) salvo em {pure_path}")
    
    # =========================
    # 2) Gráfico com Clusters
    # =========================
    # Aplica KMeans
    unique_classes = data_copy['syndrome_id'].unique()
    k = len(unique_classes)
    kmeans = KMeans(n_clusters=k, random_state=seed)
    clusters = kmeans.fit_predict(embeddings_2d)
    data_copy['cluster'] = clusters
    
    plt.figure(figsize=(12, 10))
    groups = data_copy.groupby('syndrome_id')
    for name, group in groups:
        plt.scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    
    # Plotando os centroides dos clusters
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, c="black", label="Centroides")
    
    plt.title("t-SNE de Embeddings Padronizados + Clustering (Centroides Exibidos)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Syndrome ID / Centroides", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    clusters_path = os.path.join(output_dir, output_file.replace(".png", "_clusters.png"))
    plt.savefig(clusters_path, dpi=300)
    plt.close()
    print(f"Gráfico t-SNE (com clusters) salvo em {clusters_path}")
