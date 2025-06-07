from utils.s3_utils import read_json_from_s3, upload_df_to_s3
import pandas as pd
import json

bucket = "decision-data-lake"
key = "raw/applications.json"

raw_data = json.loads(read_json_from_s3(bucket, key))

# Conversão para DataFrame, exemplo básico
df = pd.DataFrame.from_dict(raw_data, orient="index")

# Feature engineering: transforme strings, crie colunas, normalizações etc.

upload_df_to_s3(df, bucket, "features/applications_features.csv")
