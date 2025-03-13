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

def load_and_preprocess_data(picke_file):
    """
    Carrega o arquivo pickle e realiza o flatten da estrutura hierárquica.
    
    Estrutura dos dados:
    
    {
        'syndrome_id':{
            'subject_id':{
                'image_id': [320-dimensional embedding]
            }
        }
    }
    
    Retorna um DataFrame com as colunas:
    
    ['syndrome_id', 'subject_id', 'image_id', 'embedding']
    
    """
    issues = []
    
    # Abrindo o arquivo pickle fazendo a leitura do arquivo
    with open(picke_file, 'rb') as f:
        data = pickle.load(f)
        
    # Flatten da estrutura hierárquica
    records = []
    
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                # Verifica se o embedding tem o tamanho correto
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
                        'embedding': np.array(embedding)  # Converte para array numpy
                    })                    
                
    
    df = pd.DataFrame(records)
    
    return df, issues            


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

def create_balanced_dataset(df, counts, max_factor=2.0, quantile_threshold=0.7):
    """
    Cria um conjunto balanceado com base na proximidade dos embeddings ao centróide de cada classe.
    
    Para cada classe:
      - Se o número de amostras for menor ou igual ao mínimo (min_count), mantém todas as amostras.
      - Se for maior:
          1. Calcula o centróide dos embeddings da classe.
          2. Computa a distância Euclidiana de cada amostra até o centróide.
          3. Define um limiar (por exemplo, a mediana dessas distâncias, controlada pelo parâmetro quantile_threshold).
          4. Seleciona as amostras com distância menor ou igual ao limiar.
          5. Se o número de amostras selecionadas for inferior a min_count, completa com as amostras mais próximas, até atingir min_count.
          6. Se o número de amostras selecionadas for superior a max_count (definido como max_factor * min_count), então limita àquelas com as menores distâncias.
    
    Retorna:
       balanced_df: DataFrame resultante com um número balanceado de amostras por classe,
       balanced_classes: Lista das síndromes cuja quantidade original já era igual a min_count,
       imbalanced_classes: Lista das síndromes com quantidade acima de min_count.
    """
    min_count = counts.min()
    max_count = int(min_count * max_factor)
    balanced_samples = []
    
    for syndrome, group in df.groupby('syndrome_id'):
        X = np.vstack(group['embedding'].values)
        n = len(group)
        if n <= min_count:
            balanced_samples.append(group)
        else:
            # Calcula o centróide da classe
            centroid = X.mean(axis=0)
            # Calcula a distância Euclidiana de cada amostra ao centróide
            distances = np.linalg.norm(X - centroid, axis=1)
            group = group.copy()
            group['distance'] = distances
            # Define o limiar: amostras abaixo do quantil definido (ex: mediana)
            threshold = np.quantile(distances, quantile_threshold)
            selected = group[group['distance'] <= threshold]
            # Se selecionou menos que min_count, complete com as mais próximas
            if len(selected) < min_count:
                selected = group.sort_values(by='distance').iloc[:min_count]
            # Se selecionar mais que o teto, limite ao máximo permitido
            elif len(selected) > max_count:
                selected = group.sort_values(by='distance').iloc[:max_count]
            balanced_samples.append(selected)
    
    balanced_df = pd.concat(balanced_samples, axis=0).reset_index(drop=True)
    if 'distance' in balanced_df.columns:
        balanced_df.drop(columns=['distance'], inplace=True)
    
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


# =======================
# Função adicional: Análise via PCA e Clustering
# =======================
def additional_data_analysis(df, output_dir="outputs", seed=42):
    """
    Realiza análise adicional:
     - Aplica PCA para reduzir os embeddings para 2 dimensões.
     - Gera scatter plot das componentes principais coloridas por síndrome.
     - Gera boxplots e histogramas das principais componentes.
     - Realiza clustering com K-means (n_clusters = número de síndromes) e gera matriz de confusão entre clusters e classes.
     - Identifica outliers em cada síndrome usando o método IQR nas componentes principais.
    """
    
    # Filtra para as 10 síndromes mais frequentes
    counts = df.groupby('syndrome_id').size().sort_values(ascending=False)
    top_10_syndromes = counts.head(10).index.tolist()
    df_top = df[df['syndrome_id'].isin(top_10_syndromes)].copy()
    print(f"Analisando as top 10 síndromes: {top_10_syndromes}")
    
    # Extração dos embeddings e rótulos
    embeddings = np.vstack(df['embedding'].values)
    labels = df['syndrome_id'].values

    # Aplica PCA
    pca = PCA(n_components=320, random_state=seed)
    pca_result = pca.fit_transform(embeddings)
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]

    # Scatter plot dos componentes principais
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="pca1", y="pca2", hue="syndrome_id", data=df, palette="tab10", s=40, alpha=0.7)
    plt.title("PCA dos Embeddings (2 Componentes Principais)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    pca_scatter_file = os.path.join(output_dir, "pca_scatter.png")
    plt.savefig(pca_scatter_file, dpi=300)
    plt.close()
    print(f"Scatter plot PCA salvo em {pca_scatter_file}")

    # Boxplot e histogramas para pca1 por síndrome
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="syndrome_id", y="pca1", data=df, palette="Set3", hue="syndrome_id")
    plt.title("Distribuição de PCA1 por Síndrome")
    plt.xticks(rotation=45)
    plt.tight_layout()
    boxplot_file = os.path.join(output_dir, "pca1_boxplot.png")
    plt.savefig(boxplot_file, dpi=300)
    plt.close()
    print(f"Boxplot de PCA1 salvo em {boxplot_file}")
    
    # Boxplot e histogramas para pca2 por síndrome
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="syndrome_id", y="pca2", data=df, palette="Set3", hue="syndrome_id")
    plt.title("Distribuição de PCA1 por Síndrome")
    plt.xticks(rotation=45)
    plt.tight_layout()
    boxplot_file = os.path.join(output_dir, "pca2_boxplot.png")
    plt.savefig(boxplot_file, dpi=300)
    plt.close()
    print(f"Boxplot de PCA1 salvo em {boxplot_file}")

    # Histograma geral de PCA1 e PCA2
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['pca1'], bins=30, kde=True, color="steelblue")
    plt.title("Histograma de PCA1")
    plt.subplot(1, 2, 2)
    sns.histplot(df['pca2'], bins=30, kde=True, color="salmon")
    plt.title("Histograma de PCA2")
    plt.tight_layout()
    hist_file = os.path.join(output_dir, "pca_histograms.png")
    plt.savefig(hist_file, dpi=300)
    plt.close()
    print(f"Histogramas de PCA1 e PCA2 salvos em {hist_file}")
    
    # Identificação de outliers por síndrome usando IQR para PCA1
    outlier_report = {}
    for syndrome in np.unique(labels):
        syndrome_data = df[df['syndrome_id'] == syndrome]
        Q1 = syndrome_data['pca1'].quantile(0.25)
        Q3 = syndrome_data['pca1'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = syndrome_data[(syndrome_data['pca1'] < lower_bound) | (syndrome_data['pca1'] > upper_bound)]
        outlier_report[syndrome] = len(outliers)
    
    outlier_df = pd.DataFrame(list(outlier_report.items()), columns=["Syndrome", "Número de Outliers"])
    outlier_file = os.path.join(output_dir, "pca_outliers_report.csv")
    outlier_df.to_csv(outlier_file, index=False)
    print("Relatório de outliers (baseado em PCA1) salvo em", outlier_file)
    
    return df, outlier_df