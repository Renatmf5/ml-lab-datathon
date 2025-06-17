import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from stop_words import get_stop_words
from bert_vectorizer import BERTVectorizer

# Stopwords para limpeza opcional (não serão utilizadas diretamente pelo BERT)
pt_stop_words = get_stop_words('portuguese')

# 1. Carrega o dataset
data_file = os.path.join("models", "Modelo_Matching_Classificacao", "data", "matching_dataset.parquet")
df = pd.read_parquet(data_file)

# 2. Define as colunas
categorical_cols = [
    "sexo", "estado_civil", "pcd", "vaga_especifica_para_pcd", "pais_vaga",
    "nivel_academico", "tipo_contratacao", "cidade", "cidade_vaga",
    "nivel_profissional", "nivel_profissional_vaga", "ingles", "espanhol",
    "outros_idiomas", "nivel_ingles_vaga", "nivel_espanhol_vaga"
]

text_cols = [
    "titulo_profissional", "titulo_vaga", "conhecimentos_tecnicos", "certificacoes",
    "outras_certificacoes", "area_atuacao", "areas_atuacao_vaga",
    "competencia_tecnicas_e_comportamentais", "cv_candidato"
]

# 3. Define o pré-processador utilizando ColumnTransformer sem shared_model
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("bert_cv", BERTVectorizer(), "cv_candidato"),
        ("bert_title", BERTVectorizer(), "titulo_profissional"),
        ("bert_vaga", BERTVectorizer(), "titulo_vaga"),
        ("bert_tech", BERTVectorizer(), "conhecimentos_tecnicos"),
        ("bert_cert", BERTVectorizer(), "certificacoes"),
        ("bert_ocert", BERTVectorizer(), "outras_certificacoes"),
        ("bert_area", BERTVectorizer(), "area_atuacao"),
        ("bert_varea", BERTVectorizer(), "areas_atuacao_vaga"),
        ("bert_comp", BERTVectorizer(), "competencia_tecnicas_e_comportamentais"),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# 4. Define X e y
X = df[categorical_cols + text_cols]
y = df["match"]

# Calcula o scale_pos_weight para classes desbalanceadas
num_negatives = (y == 0).sum()
num_positives = (y == 1).sum()
scale_pos_weight = num_negatives / num_positives
print("Valor de scale_pos_weight:", scale_pos_weight)

# 5. Define o classificador XGBoost
classifier = XGBClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    verbosity=1,
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

# 6. Cria o pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# 7. Divide os dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 8. Treina o modelo
pipeline.fit(X_train, y_train)

# 9. Avalia o modelo treinado
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Salva o modelo treinado
joblib.dump(pipeline, "model_matching_best_bert.pkl")