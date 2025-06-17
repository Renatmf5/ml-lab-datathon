import os
import boto3
import pandas as pd
import json
from io import StringIO, BytesIO
from boto3.s3.transfer import S3Transfer, TransferConfig
import tempfile
import tqdm

s3 = boto3.client("s3")

def read_json_from_s3(bucket: str, key: str) -> dict:
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    return json.loads(content)

def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(content))

def read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(response["Body"].read()))

class ProgressPercentage:
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._tqdm = tqdm.tqdm(total=self._size, unit='B', unit_scale=True, desc=os.path.basename(filename))
    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._tqdm.update(bytes_amount)

def upload_df_to_s3(df: pd.DataFrame, bucket: str, key: str):
    # Salva o dataframe temporariamente em formato Parquet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
        temp_file_name = tmp.name
        df.to_parquet(temp_file_name, index=False)
    
    config = TransferConfig(
        multipart_threshold=1024 * 1024 * 5,  # 5 MB
        max_concurrency=10,
        multipart_chunksize=1024 * 1024 * 5,  # 5 MB
        use_threads=True
    )
    transfer = S3Transfer(s3, config)
    progress = ProgressPercentage(temp_file_name)
    print(f"Iniciando upload para s3://{bucket}/{key}")
    transfer.upload_file(temp_file_name, bucket, key, callback=progress)
    os.remove(temp_file_name)