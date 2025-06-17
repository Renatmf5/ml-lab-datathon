from utils.s3_utils import read_json_from_s3, read_parquet_from_s3
import json
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "raw"

def _ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)

def _load_with_cache(bucket: str, s3_key: str, filename: str):
    _ensure_cache_dir()
    local_path = CACHE_DIR / filename

    # se já existe em cache, carrega local
    if local_path.exists():
        with open(local_path, "r") as f:
            return json.load(f)

    # caso contrário baixa do S3 e salva no cache
    data = read_json_from_s3(bucket, s3_key)
    with open(local_path, "w") as f:
        json.dump(data, f, indent=2)
    return data

def load_applicants(bucket: str) -> dict:
    return _load_with_cache(bucket, "raw/applicants.json", "applicants.json")

def load_prospects(bucket: str) -> dict:
    return _load_with_cache(bucket, "raw/prospects.json", "prospects.json")

def load_vagas(bucket: str) -> dict:
    return _load_with_cache(bucket, "raw/vagas.json", "vagas.json")

def load_parquet_from_s3(bucket: str, s3_key: str, filename: str) -> pd.DataFrame:
    _ensure_cache_dir()
    local_path = CACHE_DIR / filename

    if local_path.exists():
        print(f"Carregando Parquet em cache: {local_path}")
        return pd.read_parquet(local_path)

    print(f"Baixando Parquet do S3: s3://{bucket}/{s3_key}")
    df = read_parquet_from_s3(bucket, s3_key)
    df.to_parquet(local_path, index=False)
    return df