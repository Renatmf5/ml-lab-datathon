import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import os
import torch
from data.uploader import save_binary_file, update_latest_txt
from data.downloader import load_parquet_from_s3


BUCKET = "decision-data-lake"
# Alterar a key para usar o arquivo processado
PROCESSED_KEY = "processed/recommendation_pairs.parquet"
# Use um nome para cache local (pode ficar na pasta raw ou outro local)
LOCAL_FILENAME = "recommendation_pairs.parquet"


EMBEDDINGS_PATH = "models/Modelo_Recomendacao_Vagas/embeddings"
INDEX_PATH = "models/Modelo_Recomendacao_Vagas/annoy_index.ann"
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

BUCKET = "decision-data-lake"

def run(models_version=None):
    # Modelo BERT
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu" if torch.cuda.is_available() else "cpu")

    # Lê os pares do datalake s3
    df = load_parquet_from_s3(BUCKET, PROCESSED_KEY, LOCAL_FILENAME)
    
    # Verifica se com log se esta executando com cuda ou cpu
    if torch.cuda.is_available():
        print("CUDA disponível, usando GPU para embeddings.")
    else:
        print("CUDA não disponível, usando CPU para embeddings.")   


    # Concatena campos de entrada
    #df["texto_candidato"] = df["cv_candidato"].fillna('') + " " + df["conhecimentos_tecnicos"].fillna('') + " " + df["titulo_profissional"].fillna('')
    #df["texto_vaga"] = df["titulo_vaga"].fillna('') + " " + df["competencia_tecnicas_e_comportamentais"].fillna('') + " " + df["area_atuacao_vaga"].fillna('')

    # Gera embeddings
    print("Gerando embeddings...")
    cand_embeddings = model.encode(df["texto_candidato"].tolist(), show_progress_bar=True)
    vaga_embeddings = model.encode(df["texto_vaga"].tolist(), show_progress_bar=True)

    # Salva numpy
    candidate_emb_file = os.path.join(EMBEDDINGS_PATH, "candidate_embeddings.npy")
    job_emb_file = os.path.join(EMBEDDINGS_PATH, "job_embeddings.npy")
    np.save(candidate_emb_file, cand_embeddings)
    np.save(job_emb_file, vaga_embeddings)

    # Cria índice Annoy
    print("Indexando vagas com Annoy...")
    dim = vaga_embeddings.shape[1]
    annoy_index = AnnoyIndex(dim, "angular")

    for i, emb in enumerate(vaga_embeddings):
        annoy_index.add_item(i, emb)

    annoy_index.build(10)
    annoy_index.save(INDEX_PATH)
    print("Index Annoy salvo em:", INDEX_PATH)
    
    # Faz upload dos arquivos para o S3
    save_binary_file(candidate_emb_file, BUCKET, f"models/Modelo_Recomendacao_Vagas/{models_version}/candidate_embeddings.npy")
    save_binary_file(job_emb_file, BUCKET, f"models/Modelo_Recomendacao_Vagas/{models_version}/job_embeddings.npy")
    save_binary_file(INDEX_PATH, BUCKET, f"models/Modelo_Recomendacao_Vagas/{models_version}/annoy_index.ann")

    # Atualiza o arquivo latest.txt com todos os arquivos do modelo
    latest_key = f"models/Modelo_Recomendacao_Vagas/latest.txt"
    latest_content = f"{models_version}/candidate_embeddings.npy\n" \
                   f"{models_version}/job_embeddings.npy\n" \
                   f"{models_version}/annoy_index.ann\n"
    update_latest_txt(BUCKET, latest_key, latest_content)
