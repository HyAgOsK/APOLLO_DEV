import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


# 1) Carrega o arquivo pickle
with open('/home/linux/APOLLO_TEST/data/mini_gm_public_v0.1.p', 'rb') as f:
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

print(df_original.head())

print(df_original.describe())

print(df_original['embedding'].apply(len).value_counts())

embeddings = np.vstack(df_original['embedding'].to_numpy())  # matriz (n_amostras, n_features)


emb_list = []
# Exemplo para duas classes
for syndrome_id, subjects in data.items():
    emb_list.append(embeddings[df_original['syndrome_id'] == syndrome_id]) 
    # Distância média entre elas
    if len(emb_list) > 1:
        dist = cdist(emb_list[-1], emb_list[-2]).mean()
        print(f"Distância média entre {syndrome_id} e {list(data.keys())[list(data.keys()).index(syndrome_id)-1]}:", dist)
    else:
        print(f"Distância média entre {syndrome_id} e a primeira classe: Não aplicável")

