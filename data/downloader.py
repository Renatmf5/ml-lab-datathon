import os
import json
import pickle  # Para eventual uso
from datetime import datetime
import pandas as pd
from pathlib import Path
import boto3

from utils.s3_utils import read_json_from_s3, read_parquet_from_s3

s3 = boto3.client("s3")
CACHE_DIR = Path(__file__).parent / "raw"

def _ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)

def get_s3_file_last_modified(bucket: str, key: str) -> datetime:
    response = s3.head_object(Bucket=bucket, Key=key)
    return response['LastModified']

def _load_with_cache(bucket: str, s3_key: str, filename: str) -> dict:
    """
    Carrega um JSON do cache se estiver atualizado; caso contrário, baixa do S3,
    atualiza o cache e os metadados.
    """
    _ensure_cache_dir()
    local_file_path = CACHE_DIR / filename
    metadata_file_path = CACHE_DIR / f"{filename}_metadata.txt"
    
    s3_last_modified = get_s3_file_last_modified(bucket, s3_key)
    
    # Se o arquivo em cache e o metadado existem, verifica se estão atualizados
    if local_file_path.exists() and metadata_file_path.exists():
        with open(metadata_file_path, "r") as meta_file:
            cached_last_modified_str = meta_file.read().strip()
        try:
            cached_last_modified = datetime.fromisoformat(cached_last_modified_str)
        except Exception:
            cached_last_modified = None
        
        if cached_last_modified:
            # Converte ambas para offset-naive para comparação
            if cached_last_modified.replace(tzinfo=None) >= s3_last_modified.replace(tzinfo=None):
                print(f"Carregando JSON em cache: {local_file_path}")
                with open(local_file_path, "r") as f:
                    return json.load(f)
    
    # Caso contrário, baixa do S3 e atualiza o cache e metadados
    print(f"Baixando JSON do S3: s3://{bucket}/{s3_key}")
    data = read_json_from_s3(bucket, s3_key)
    with open(local_file_path, "w") as f:
        json.dump(data, f, indent=2)
    # Atualiza metadados com a última modificação
    with open(metadata_file_path, "w") as meta_file:
        meta_file.write(s3_last_modified.isoformat())
    return data

def load_applicants(bucket: str) -> dict:
    return _load_with_cache(bucket, "raw/applicants.json", "applicants.json")

def load_prospects(bucket: str) -> dict:
    return _load_with_cache(bucket, "raw/prospects.json", "prospects.json")

def load_vagas(bucket: str) -> dict:
    return _load_with_cache(bucket, "raw/vagas.json", "vagas.json")

def load_parquet_from_s3(bucket: str, s3_key: str, filename: str) -> pd.DataFrame:
    """
    Carrega um arquivo Parquet do cache se estiver atualizado; caso contrário, 
    baixa do S3, atualiza o cache e os metadados.
    """
    _ensure_cache_dir()
    local_file_path = CACHE_DIR / filename
    metadata_file_path = CACHE_DIR / f"{filename}_metadata.txt"

    s3_last_modified = get_s3_file_last_modified(bucket, s3_key)
    
    if local_file_path.exists() and metadata_file_path.exists():
        with open(metadata_file_path, "r") as meta_file:
            cached_last_modified_str = meta_file.read().strip()
        try:
            cached_last_modified = datetime.fromisoformat(cached_last_modified_str)
        except Exception:
            cached_last_modified = None

        if cached_last_modified:
            if cached_last_modified.replace(tzinfo=None) >= s3_last_modified.replace(tzinfo=None):
                print(f"Carregando Parquet em cache: {local_file_path}")
                return pd.read_parquet(local_file_path)

    print(f"Baixando Parquet do S3: s3://{bucket}/{s3_key}")
    df = read_parquet_from_s3(bucket, s3_key)
    df.to_parquet(local_file_path, index=False)
    # Atualiza os metadados com a data do S3
    with open(metadata_file_path, "w") as meta_file:
        meta_file.write(s3_last_modified.isoformat())
    return df