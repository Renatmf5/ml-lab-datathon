# Supondo que no seu df as informações de candidato e vaga estejam separadas ou possam ser extraídas:
import pandas as pd
from data.uploader import save_parquet
from pathlib import Path
from data.downloader import load_parquet_from_s3
from data.uploader import save_parquet

BUCKET = "decision-data-lake"
S3_KEY = "features/candidates.parquet"   # para leitura dos dados originais
LOCAL_FILENAME = "candidates.parquet"
# Caminhos

OUTPUT_PATH = Path("models/Modelo_Recomendacao_Vagas/data/recommendation_pairs.parquet")

def build():
    # Carrega os dados
    df = load_parquet_from_s3(BUCKET, S3_KEY, LOCAL_FILENAME)

    # Filtra candidatos com situações positivas
    MATCH_SITUACOES_POSITIVAS = [
        "Aprovado", "Contratado pela Decision", "Encaminhar Proposta", "Proposta Aceita",
        "Encaminhado ao Requisitante", "Entrevista com Cliente", "Entrevista Técnica"
    ]
    df = df[df["situacao_candidato"].isin(MATCH_SITUACOES_POSITIVAS)].dropna()

    # Consolida os campos para candidatos e vagas
    # (Observe que, se o df vem com informações de candidato e também de vaga, 
    #  pode ser necessário separá-los ou tratar as colunas de vaga que pertencem ao registro)
    df["texto_candidato"] = (
        df["titulo_profissional"].fillna('') + " " +
        df["conhecimentos_tecnicos"].fillna('') + " " +
        df["certificacoes"].fillna('') + " " +
        df["outras_certificacoes"].fillna('') + " " +
        df["cidade"].fillna('') + " " +  # Adiciona cidade se necessário
        df["ingles"].fillna('') + " " +
        df["espanhol"].fillna('') + " " +
        df["outros_idiomas"].fillna('') + " " + 
        df["pcd"].fillna('') + " " +
        df["cv_candidato"].fillna('')
    )
    df["texto_vaga"] = (
        df["titulo_vaga"].fillna('') + " " +
        df["competencia_tecnicas_e_comportamentais"].fillna('') + " " +
        df["areas_atuacao_vaga"].fillna('') + " " +
        df["cidade_vaga"].fillna('') + " " +  # Adiciona cidade da vaga se necessário
        df["estado_vaga"].fillna('') + " " +
        df["nivel_profissional_vaga"].fillna('') + " " +
        df["nivel_academico_vaga"].fillna('') + " " +
        df["nivel_ingles_vaga"].fillna('') + " " +
        df["nivel_espanhol_vaga"].fillna('') + " " +
        df["outros_idiomas_vaga"].fillna('') + " " +
        df["vaga_especifica_para_pcd"].fillna('')
    )

    # Se houver uma coluna que indique a cidade ou área, você pode usá-la para filtrar:
    # Por exemplo, considerando "cidade" no candidato e "cidade_vaga" na vaga:
    #df_filtrado = df[df["cidade"] == df["cidade_vaga"]].copy()

    # Se não houver, você pode optar por definir outro critério (como área de atuação ou nível profissional).

    # Opcionalmente, defina colunas importantes para manter
    cols_output = [
        "codigo", "texto_candidato", "texto_vaga",
        "titulo_profissional", "conhecimentos_tecnicos", "cv_candidato",
        "titulo_vaga", "competencia_tecnicas_e_comportamentais", "areas_atuacao_vaga",
        "cidade", "cidade_vaga"
    ]
    df_pairs = df[cols_output]

    # (Opcional) Salva localmente para debug
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_parquet(OUTPUT_PATH, index=False)

    # Salva os pares no bucket S3 na pasta processed
    s3_key_processed = "processed/recommendation_pairs.parquet"
    save_parquet(df_pairs, BUCKET, s3_key_processed)
    print(f"Pares filtrados salvos no bucket {BUCKET} com a chave {s3_key_processed}")