import joblib
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from stop_words import get_stop_words
from data.uploader import save_model_pickle, get_next_model_version, update_latest_txt
from data.downloader import load_parquet_from_s3
from imblearn.over_sampling import SMOTE

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BUCKET = "decision-data-lake"
PROCESSED_KEY = "processed/matching_dataset.parquet"
LOCAL_FILENAME = "matching_dataset.parquet"

def create_text_pipeline(max_features, n_components, combined_stop_words, ngram_range=(1,2), min_df=3):
    """Cria uma pipeline de processamento textual com TF-IDF e TruncatedSVD."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
                    max_features=max_features,
                    stop_words=combined_stop_words,
                    ngram_range=ngram_range,
                    min_df=min_df)),
        ("svd", TruncatedSVD(n_components=n_components))
    ])

def run():
    # Stopwords combinadas
    pt_stop_words = set(get_stop_words('portuguese'))
    combined_stop_words = list(pt_stop_words.union(ENGLISH_STOP_WORDS))
    
    # Carrega o dataset
    logger.info("Carregando dados do S3...")
    df = load_parquet_from_s3(BUCKET, PROCESSED_KEY, LOCAL_FILENAME)
    
    # Definição de colunas
    categorical_cols = [
        "sexo", "estado_civil", "pcd", "vaga_especifica_para_pcd", "pais_vaga",
        "nivel_academico", "nivel_academico_vaga", "tipo_contratacao", "cidade", "cidade_vaga",
        "nivel_profissional", "nivel_profissional_vaga", "ingles", "espanhol",
        "outros_idiomas", "nivel_ingles_vaga", "nivel_espanhol_vaga"
    ]
    
    primary_text_cols = ["cv_candidato", "principais_atividades"]
    secondary_text_cols = [
        "titulo_profissional", "titulo_vaga", "conhecimentos_tecnicos", 
        "area_atuacao", "areas_atuacao_vaga", "competencia_tecnicas_e_comportamentais"
    ]
    tertiary_text_cols = ["certificacoes", "outras_certificacoes"]
    text_cols = primary_text_cols + secondary_text_cols + tertiary_text_cols
    
    # Cria feature combinada para capturar relações entre habilidades e atividades
    df['combined_keywords'] = df['conhecimentos_tecnicos'].fillna('') + ' ' + df['principais_atividades'].fillna('')
    
    # Cria o pré-processador usando pipelines customizadas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            
            # Pipelines para campos textuais
            ("text_cv", create_text_pipeline(max_features=150, n_components=30, combined_stop_words=combined_stop_words,
                                              ngram_range=(1,2), min_df=3), "cv_candidato"),
            
            ("text_atividades", create_text_pipeline(max_features=150, n_components=30, combined_stop_words=combined_stop_words,
                                                       ngram_range=(1,2), min_df=3), "principais_atividades"),
            
            ("text_combined", create_text_pipeline(max_features=150, n_components=30, combined_stop_words=combined_stop_words,
                                                     ngram_range=(1,2), min_df=2), "combined_keywords"),
            
            ("text_tech", create_text_pipeline(max_features=80, n_components=20, combined_stop_words=combined_stop_words,
                                                 ngram_range=(1,1), min_df=3), "conhecimentos_tecnicos"),
            
            ("text_title", create_text_pipeline(max_features=50, n_components=15, combined_stop_words=combined_stop_words,
                                                  ngram_range=(1,1), min_df=3), "titulo_profissional"),
            
            ("text_vaga", create_text_pipeline(max_features=50, n_components=15, combined_stop_words=combined_stop_words,
                                                 ngram_range=(1,1), min_df=3), "titulo_vaga"),
            
            ("text_area", create_text_pipeline(max_features=50, n_components=15, combined_stop_words=combined_stop_words,
                                                 ngram_range=(1,1), min_df=3), "area_atuacao"),
            
            ("text_varea", create_text_pipeline(max_features=50, n_components=15, combined_stop_words=combined_stop_words,
                                                  ngram_range=(1,1), min_df=3), "areas_atuacao_vaga"),
            
            ("text_competencias", create_text_pipeline(max_features=70, n_components=20, combined_stop_words=combined_stop_words,
                                                         ngram_range=(1,1), min_df=3), "competencia_tecnicas_e_comportamentais"),
            
            ("text_cert", create_text_pipeline(max_features=40, n_components=10, combined_stop_words=combined_stop_words,
                                                 ngram_range=(1,1), min_df=3), "certificacoes"),
            
            ("text_ocert", create_text_pipeline(max_features=40, n_components=10, combined_stop_words=combined_stop_words,
                                                  ngram_range=(1,1), min_df=3), "outras_certificacoes"),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    # Define X e y
    X = df[categorical_cols + text_cols + ['combined_keywords']]
    y = df["match"]
    
    # Cálculo do scale_pos_weight para dataset desbalanceado
    num_negatives = (y == 0).sum()
    num_positives = (y == 1).sum()
    scale_pos_weight = num_negatives / num_positives
    logger.info(f"Valor de scale_pos_weight: {scale_pos_weight:.2f} (proporção negativo/positivo)")
    
    # Configuração do classificador XGBoost ajustado
    classifier = XGBClassifier(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=500,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )
    
    # Divisão dos dados
    logger.info("Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Distribuição dos dados - Treino: {y_train.value_counts().to_dict()}, Teste: {y_test.value_counts().to_dict()}")
    
    # Pré-processamento do conjunto de treino
    logger.info("Aplicando transformações ao conjunto de treino...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Aplicação de SMOTE para balanceamento das classes
    logger.info("Aplicando SMOTE para balancear as classes...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    logger.info(f"Distribuição após SMOTE - Treino: {np.bincount(y_train_resampled)}")
    
    # Treinamento do classificador
    logger.info("Treinando o classificador nos dados balanceados...")
    classifier.fit(X_train_resampled, y_train_resampled)
    
    # Transformação do conjunto de teste
    X_test_transformed = preprocessor.transform(X_test)
    
    # Avaliação do modelo com threshold padrão
    logger.info("Avaliando o modelo com threshold padrão...")
    y_pred = classifier.predict(X_test_transformed)
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Criação de um pipeline final para uso em produção
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    # Salvamento do modelo localmente
    logger.info("Salvando modelo localmente...")
    joblib.dump(final_pipeline, "model_matching_best.pkl")
    
    # Salvamento do modelo no S3 com versionamento
    bucket_name = BUCKET
    model_version = get_next_model_version(bucket_name, "models/Modelo_Matching_Classificacao")
    model_key = f"models/Modelo_Matching_Classificacao/{model_version}/model_matching.pkl"
    save_model_pickle(final_pipeline, bucket_name, model_key)
    logger.info(f"Modelo treinado e salvo em: {model_key} no bucket {bucket_name}")
    
    # Atualiza o arquivo latest.txt no S3
    latest_key = "models/Modelo_Matching_Classificacao/latest.txt"
    latest_content = f"{model_version}/model_matching.pkl"
    update_latest_txt(bucket_name, latest_key, latest_content)
    
    return model_version