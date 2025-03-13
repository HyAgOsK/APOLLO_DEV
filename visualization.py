"""
--------------------------------------------------------

                ML Junior Practical Test.

         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
         
--------------------------------------------------------
"""

# Importação das bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def plot_tsne(data_df, output_file='tsne_plot.png', output_dir='outputs_balanced_data', seed=42):
    
    """
        Reduz os embeddings para 2 dimensões usando t-SNE e 
        plota um gráfico da dispersão. Colorindo cada ponto de acordo
        com seu syndrome_id, ou sua classe respectivamente
    
    """
    
    # Empilha os embeddings em uma matriz
    embeddings = np.stack(data_df['embedding'].values)
    
    # Reduz a dimensionalidade para 2 dimensões
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=5000000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Adiciona as coordenadas 2D ao DataFrame
    data_df = data_df.copy()
    data_df['tsne_x'] = embeddings_2d[:, 0]
    data_df['tsne_y'] = embeddings_2d[:, 1]
    
    # Plota o gráfico
    plt.figure(figsize=(10, 8))
    groups = data_df.groupby('syndrome_id')
    for name, group in groups:
        plt.scatter(group['tsne_x'], group['tsne_y'], label=name, alpha=0.7, s=30)
    
    plt.title("t-SNE plot Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title='Sindrome ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    output_file = os.path.join(output_dir, "tsne_plot.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
        