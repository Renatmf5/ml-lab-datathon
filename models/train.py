import joblib
from sklearn.ensemble import RandomForestClassifier

# leitura
df = pd.read_csv("s3://decision-data-lake/features/applications_features.csv")  # via s3fs ou boto3 + pd.read_csv

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier().fit(X, y)

joblib.dump(model, "modelo.joblib")
# upload para s3: models/rf_model.joblib
