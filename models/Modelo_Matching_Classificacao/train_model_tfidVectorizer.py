import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from stop_words import get_stop_words
from data.uploader import save_model_pickle, get_next_model_version, update_latest_txt
from data.downloader import load_parquet_from_s3
from imblearn.over_sampling import SMOTE

BUCKET = "decision-data-lake"
PROCESSED_KEY = "processed/matching_dataset.parquet"
LOCAL_FILENAME = "matching_dataset.parquet"

def run():
    # Stopwords combinadas
    pt_stop_words = set(get_stop_words('portuguese'))
    combined_stop_words = list(pt_stop_words.union(ENGLISH_STOP_WORDS))
    
    # Carrega o dataset
    print("Carregando dados do S3...")
    df = load_parquet_from_s3(BUCKET, PROCESSED_KEY, LOCAL_FILENAME)
    
    # Define as colunas
    categorical_cols = [
        "sexo", "estado_civil", "pcd", "vaga_especifica_para_pcd", "pais_vaga",
        "nivel_academico","nivel_academico_vaga", "tipo_contratacao", "cidade", "cidade_vaga",
        "nivel_profissional", "nivel_profissional_vaga", "ingles", "espanhol",
        "outros_idiomas", "nivel_ingles_vaga", "nivel_espanhol_vaga"
    ]
    
    # Separamos os campos textuais por importância/tamanho
    primary_text_cols = ["cv_candidato", "principais_atividades"]
    secondary_text_cols = [
        "titulo_profissional", "titulo_vaga", "conhecimentos_tecnicos", 
        "area_atuacao", "areas_atuacao_vaga", "competencia_tecnicas_e_comportamentais"
    ]
    tertiary_text_cols = ["certificacoes", "outras_certificacoes"]
    
    # Adicionamos todos ao text_cols para processamento conjunto
    text_cols = primary_text_cols + secondary_text_cols + tertiary_text_cols
    
    # Criamos um novo recurso combinando informações do candidato e da vaga
    df['combined_keywords'] = df['conhecimentos_tecnicos'].fillna('') + ' ' + df['principais_atividades'].fillna('')
    
    # Pré-processador com configurações otimizadas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            
            # Campos de texto primários (grandes) - mais componentes e features
            ("text_cv", Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=150, 
                    stop_words=combined_stop_words,
                    ngram_range=(1, 2),  # Incluímos bigrams
                    min_df=3,            # Mínimo de ocorrências
                )),
                ("svd", TruncatedSVD(n_components=30))
            ]), "cv_candidato"),
            
            ("text_atividades", Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=150, 
                    stop_words=combined_stop_words,
                    ngram_range=(1, 2),
                    min_df=3,
                )),
                ("svd", TruncatedSVD(n_components=30))
            ]), "principais_atividades"),
            
            # Feature combinada para capturar relações entre habilidades e atividades
            ("text_combined", Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=150, 
                    stop_words=combined_stop_words,
                    ngram_range=(1, 2),
                    min_df=2,
                )),
                ("svd", TruncatedSVD(n_components=30))
            ]), "combined_keywords"),
            
            # Campos secundários
            ("text_tech", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=80, stop_words=combined_stop_words)),
                ("svd", TruncatedSVD(n_components=20))
            ]), "conhecimentos_tecnicos"),
            
            ("text_title", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=50, stop_words=combined_stop_words)),
                ("svd", TruncatedSVD(n_components=15))
            ]), "titulo_profissional"),
            
            ("text_vaga", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=50, stop_words=combined_stop_words)),
                ("svd", TruncatedSVD(n_components=15))
            ]), "titulo_vaga"),
            
            ("text_area", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=50, stop_words=combined_stop_words)),
                ("svd", TruncatedSVD(n_components=15))
            ]), "area_atuacao"),
            
            ("text_varea", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=50, stop_words=combined_stop_words)),
                ("svd", TruncatedSVD(n_components=15))
            ]), "areas_atuacao_vaga"),
            
            ("text_competencias", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=70, stop_words=combined_stop_words)),
                ("svd", TruncatedSVD(n_components=20))
            ]), "competencia_tecnicas_e_comportamentais"),
            
            # Campos terciários
            ("text_cert", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=40, stop_words=combined_stop_words)),
                ("svd", TruncatedSVD(n_components=10))
            ]), "certificacoes"),
            
            ("text_ocert", Pipeline([
                ("tfidf", TfidfVectorizer(max_features=40, stop_words=combined_stop_words)),
                ("svd", TruncatedSVD(n_components=10))
            ]), "outras_certificacoes"),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    # Define X e y
    # Incluímos a feature combinada
    X = df[categorical_cols + text_cols + ['combined_keywords']]
    y = df["match"]
    
    # Cálculo do scale_pos_weight para dataset desbalanceado
    num_negatives = (y == 0).sum()
    num_positives = (y == 1).sum()
    scale_pos_weight = num_negatives / num_positives
    print(f"Valor de scale_pos_weight: {scale_pos_weight} (proporção negativo/positivo)")
    
    # Classificador XGBoost ajustado
    classifier = XGBClassifier(
        learning_rate=0.05,         # Reduzido para permitir convergência mais suave
        max_depth=5,                # Mantido em 5 conforme melhoria anterior
        n_estimators=500,           # Aumentado para compensar o learning rate menor
        min_child_weight=3,         # Ajuda a reduzir overfitting
        subsample=0.8,              # Usa 80% dos dados em cada árvore
        colsample_bytree=0.8,       # Usa 80% das features em cada árvore
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )
    
    # Divisão dos dados
    print("Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Distribuição dos dados - Treino: {y_train.value_counts()}, Teste: {y_test.value_counts()}")
    
    # Pré-processamento dos dados de treino
    print("Aplicando transformações ao conjunto de treino...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Aplicando SMOTE para balancear as classes...
    print("Aplicando SMOTE para balancear as classes...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)  # Aumenta minoria para 30% da maioria
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    
    print(f"Distribuição após SMOTE - Treino: {np.bincount(y_train_resampled)}")
    
    # Treinamento do classificador nos dados balanceados
    print("Treinando o classificador nos dados balanceados...")
    classifier.fit(X_train_resampled, y_train_resampled)
    
    # Transforma dados de teste
    X_test_transformed = preprocessor.transform(X_test)
    
    # Avaliação com threshold padrão
    print("Avaliando modelo com threshold padrão...")
    y_pred = classifier.predict(X_test_transformed)
    print(classification_report(y_test, y_pred))
    
    # Criando um pipeline apenas para salvar - não usamos no treinamento acima 
    # porque o SMOTE não funcionaria dentro do pipeline normal
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    # Salva o modelo localmente
    print("Salvando modelo localmente...")
    joblib.dump(final_pipeline, "model_matching_best.pkl")
    
    # Salva o modelo no S3
    bucket_name = BUCKET
    model_version = get_next_model_version(bucket_name, "models/Modelo_Matching_Classificacao")
    model_key = f"models/Modelo_Matching_Classificacao/{model_version}/model_matching.pkl"
    save_model_pickle(final_pipeline, bucket_name, model_key)
    print(f"Modelo treinado e salvo em: {model_key} no bucket {bucket_name}")
    
    # Atualiza o arquivo latest.txt no S3
    latest_key = "models/Modelo_Matching_Classificacao/latest.txt"
    latest_content = f"{model_version}/model_matching.pkl"
    update_latest_txt(bucket_name, latest_key, latest_content)

    return model_version