from utils.s3_utils import read_json_from_s3, upload_df_to_s3
import pandas as pd

BUCKET = "decision-data-lake"
RAW_KEY = "raw/applicants.json"
FEATURES_KEY = "features/teste_features.csv"

# Leitura JSON
data = read_json_from_s3(BUCKET, RAW_KEY)
print("JSON carregado:", data)

# Exemplo: transforma em DataFrame
df = pd.DataFrame.from_dict(data)
print("DataFrame:", df.head())

# Upload CSV no bucket/features
upload_df_to_s3(df, BUCKET, FEATURES_KEY)
print("Upload feito!")
