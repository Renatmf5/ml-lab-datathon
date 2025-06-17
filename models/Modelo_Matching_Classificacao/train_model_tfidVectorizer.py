import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from stop_words import get_stop_words
from data.uploader import save_model_pickle, get_next_model_version, update_latest_txt
from data.downloader import load_parquet_from_s3

BUCKET = "decision-data-lake"
# Alterar a key para usar o arquivo processado
PROCESSED_KEY = "processed/matching_dataset.parquet"
# Use um nome para cache local (pode ficar na pasta raw ou outro local)
LOCAL_FILENAME = "matching_dataset.parquet"

def run():
    # Stopwords em português
    pt_stop_words = get_stop_words('portuguese')
    
    # Carrega o dataset diretamente do S3 (usando cache se existir)
    df = load_parquet_from_s3(BUCKET, PROCESSED_KEY, LOCAL_FILENAME)
    
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
    
    # 3. Define o pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("text_title", TfidfVectorizer(max_features=20, stop_words=pt_stop_words), "titulo_profissional"),
            ("text_tech", TfidfVectorizer(max_features=20, stop_words=pt_stop_words), "conhecimentos_tecnicos"),
            ("text_cert", TfidfVectorizer(max_features=50, stop_words=pt_stop_words), "certificacoes"),
            ("text_ocert", TfidfVectorizer(max_features=50, stop_words=pt_stop_words), "outras_certificacoes"),
            ("text_area", TfidfVectorizer(max_features=50, stop_words=pt_stop_words), "area_atuacao"),
            ("text_varea", TfidfVectorizer(max_features=20, stop_words=pt_stop_words), "areas_atuacao_vaga"),
            ("text_competencias", TfidfVectorizer(max_features=50, stop_words=pt_stop_words), "competencia_tecnicas_e_comportamentais"),
            ("text_cv", TfidfVectorizer(max_features=100, stop_words=pt_stop_words), "cv_candidato")
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    # 4. Define X e y
    X = df[categorical_cols + text_cols]
    y = df["match"]
    
    # Calcula scale_pos_weight
    num_negatives = (y == 0).sum()
    num_positives = (y == 1).sum()
    scale_pos_weight = num_negatives / num_positives
    print("Valor de scale_pos_weight:", scale_pos_weight)
    
    # 5. Define o classificador
    classifier = XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
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
    
    # 9. Avalia o modelo
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 10. Salva o modelo treinado localmente
    joblib.dump(pipeline, "model_matching_best.pkl")
    
    # 11. Salva o modelo no S3
    bucket_name = BUCKET
    model_version = get_next_model_version(bucket_name, "models/Modelo_Matching_Classificacao")
    model_key = f"models/Modelo_Matching_Classificacao/{model_version}/model_matching.pkl"
    save_model_pickle(pipeline, bucket_name, model_key)
    print(f"Modelo treinado e salvo em: {model_key} no bucket {bucket_name}")
    
    # 12. Atualiza o arquivo latest.txt no S3
    latest_key = "models/Modelo_Matching_Classificacao/latest.txt"
    latest_content = f"{model_version}/model_matching.pkl"
    update_latest_txt(bucket_name, latest_key, latest_content)

    return model_version
    