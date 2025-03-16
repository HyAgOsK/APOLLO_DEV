"""
--------------------------------------------------------

                ML Junior Practical Test.

         @Autor: HYAGO VIEIRA LEMES BAROSA SILVA
         
--------------------------------------------------------
"""
# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE


def plot_syndrome_distribution(df, title="Distribuição de imagens por síndrome", name=None, output_dir="images_per_syndrome", output_file="syndrome_distribution.png"):
    """ Função para plotar a distribuição de imagens por síndrome """
    counts = df.groupby('syndrome_id').size().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    counts.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel("Syndrome ID")
    plt.ylabel("Número de amostras")
    plt.tight_layout()
    final_path = os.path.join(output_dir, output_file) 
    plt.savefig(final_path, dpi=300)
    plt.close()
    
def smote_augmentation(df, random_state=0, k_neighbors=5):
    """
    Aplica SMOTE para criar embeddings sintéticos das classes minoritárias.
    Retorna um novo DataFrame balanceado.
    """
    X = np.stack(df['embedding'].values) 
    y = df['syndrome_id'].values
    
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    
    data_smote = []
    for i, (emb, label) in enumerate(zip(X_res, y_res)):
        data_smote.append({
            'syndrome_id': label,
            'subject_id': f'smote_subject_{i}',
            'image_id': f'smote_image_{i}',
            'embedding': emb
        })
    
    df_smote = pd.DataFrame(data_smote)
    return df_smote

def noise_augmentation(df, noise_std=0.01, n_new_per_sample=1):
    """
    Aplica uma perturbação gaussiana (ruído) de desvio padrão noise_std
    em cada embedding, gerando 'n_new_per_sample' novas amostras por exemplo.
    Retorna df original + sintéticos.
    """
    augmented_rows = []
    for idx, row in df.iterrows():
        original_embedding = row['embedding']
        
        for i in range(n_new_per_sample):
            noisy_embedding = original_embedding + np.random.normal(
                0, noise_std, size=original_embedding.shape
            )
            new_row = row.copy()
            new_row['subject_id'] = f"{row['subject_id']}_noisy_{i}"
            new_row['image_id'] = f"{row['image_id']}_noisy_{i}"
            new_row['embedding'] = noisy_embedding
            augmented_rows.append(new_row)
    
    df_noisy = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    return df_noisy

def mixup_augmentation(df, n_samples=1000, alpha=0.5, random_state=0):
    """
    Gera 'n_samples' novos embeddings por Mixup, escolhendo pares aleatórios 
    da mesma síndrome e interpolando (alpha e 1-alpha).
    """
    rnd = np.random.RandomState(random_state)
    
    grouped = df.groupby('syndrome_id')
    
    mixup_data = []
    for _ in range(n_samples):
        syndrome_id = rnd.choice(df['syndrome_id'].unique())
        group = grouped.get_group(syndrome_id)
        
        row1 = group.sample(1, random_state=rnd).iloc[0]
        row2 = group.sample(1, random_state=rnd).iloc[0]
        
        emb1 = row1['embedding']
        emb2 = row2['embedding']
        
        new_emb = alpha * emb1 + (1 - alpha) * emb2
        
        mixup_data.append({
            'syndrome_id': syndrome_id,
            'subject_id': f"mixup_subj_{rnd.randint(100000)}",
            'image_id': f"mixup_img_{rnd.randint(100000)}",
            'embedding': new_emb
        })
    
    df_mixup_new = pd.DataFrame(mixup_data)
    df_mixup = pd.concat([df, df_mixup_new], ignore_index=True)
    return df_mixup
