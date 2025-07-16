from pathlib import Path
from data.downloader import load_parquet_from_s3
from data.uploader import save_parquet  

BUCKET = "decision-data-lake"
BUCKET_FEATURES = "decision-data-lake-features"
S3_KEY = "features/candidates.parquet"   # para leitura dos dados originais
LOCAL_FILENAME = "candidates.parquet"
# Defina a key para o processed dataset no S3
PROCESSED_KEY = "processed/matching_dataset.parquet"

OUTPUT_PATH = Path("models/Modelo_Matching_Classificacao/data/matching_dataset.parquet")

MATCH_SITUACOES_POSITIVAS = [
    "Aprovado", "Contratado pela Decision", "Encaminhar Proposta", "Proposta Aceita"
]

def build():
    print(f"Lendo features de: s3://{BUCKET_FEATURES}/{S3_KEY}")
    df = load_parquet_from_s3(BUCKET_FEATURES, S3_KEY, LOCAL_FILENAME)
    print(f"Total de registros lidos: {len(df)}")
    
    df["match"] = df["situacao_candidato"].isin(MATCH_SITUACOES_POSITIVAS).astype(int)
    selected_columns = [
        "match",
        "sexo", "estado_civil", "pcd", "cidade", "titulo_profissional", "area_atuacao",
        "conhecimentos_tecnicos", "certificacoes", "outras_certificacoes",
        "remuneracao", "nivel_profissional", "nivel_academico",
        "ingles", "espanhol", "outros_idiomas", "cv_candidato",
        "titulo_vaga", "tipo_contratacao", "prazo_contratacao", "prioridade_vaga",
        "pais_vaga", "estado_vaga", "cidade_vaga", "vaga_especifica_para_pcd",
        "nivel_profissional_vaga", "nivel_academico_vaga", "nivel_ingles_vaga",
        "nivel_espanhol_vaga", "outros_idiomas_vaga", "areas_atuacao_vaga", "competencia_tecnicas_e_comportamentais","principais_atividades",
    ]
    df_filtered = df[selected_columns].dropna(subset=["match"])
    print(f"Registros após filtro de nulos em match: {len(df_filtered)}")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(OUTPUT_PATH, index=False)
    print("\nMatching dataset salvo localmente em:", OUTPUT_PATH)
    
    # Salva o dataset processado no S3 na pasta "processed/"
    save_parquet(df_filtered, BUCKET, PROCESSED_KEY)
    print(f"Matching dataset também salvo em s3://{BUCKET}/{PROCESSED_KEY}")
    print("Pronto para treinar o modelo!")
    
