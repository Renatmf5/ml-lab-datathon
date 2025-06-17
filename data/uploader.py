from utils.s3_utils import upload_df_to_s3
import boto3
import pickle
import io
import os

def save_features(df, bucket: str, name: str):
    key = f"features/{name}.parquet"
    upload_df_to_s3(df, bucket, key)
    
def save_parquet(df, bucket: str, key: str):
    upload_df_to_s3(df, bucket, key)

def save_model_pickle(model, bucket: str, key: str):
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    
def save_model_pkl(model, bucket: str, key: str):
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

def save_binary_file(file_path: str, bucket: str, key: str):
    """
    Lê um arquivo binário (por exemplo, .npy ou .ann) e faz o upload para o S3.
    :param file_path: Caminho local do arquivo.
    :param bucket: Nome do bucket no S3.
    :param key: Caminho/Chave no bucket onde o arquivo será salvo.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")
        
    with open(file_path, "rb") as f:
        data = f.read()
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"Arquivo {file_path} salvo em s3://{bucket}/{key}")

def get_next_model_version(bucket, model_prefix):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{model_prefix}/"
    versions = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            folder = common_prefix["Prefix"].split("/")[-2]
            if folder.startswith("v") and folder[1:].isdigit():
                versions.append(int(folder[1:]))
    next_version = max(versions) + 1 if versions else 1
    return f"v{next_version}"

# ...existing code...

def update_latest_txt(bucket: str, key: str, content: str):
    """
    Atualiza ou cria o arquivo latest.txt no S3 com o conteúdo informado.
    :param bucket: Nome do bucket no S3.
    :param key: Caminho/Chave do arquivo (por exemplo, "models/Modelo_Matching_Classificacao/latest.txt").
    :param content: Conteúdo a ser escrito no arquivo.
    """
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=content)
    print(f"Arquivo {key} atualizado com o conteúdo: {content}")