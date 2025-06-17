import numpy as np
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import pandas as pd

# Caminhos
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = "models/Modelo_Recomendacao_Vagas/annoy_index.ann"
EMBEDDINGS_PATH = "models/Modelo_Recomendacao_Vagas/embeddings/job_embeddings.npy"
PAIRS_PATH = "models/Modelo_Recomendacao_Vagas/data/recommendation_pairs.parquet"

def generate_recommendations():
    # Carrega
    model = SentenceTransformer(MODEL)
    annoy_index = AnnoyIndex(384, "angular")
    annoy_index.load(INDEX_PATH)
    job_embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_parquet(PAIRS_PATH)
    """
    Profissional com experiência em SAP-FI, SAP-MM e controladoria.
    Atuação com análise de negócios, parametrizações e suporte técnico.
    Conhecimento em Visual Studio e SQL Server.
    """
    # Input do candidato
    cv_text = """
    Profissional com experiência em informatica powercenter, informatica MDM e com bons conhecimentos em Cloud aws.
    Atuação com  crição da Data-Lakes e Datawarehouse. conhecimento em ETL e ELT. ja atuando em mercados de telecomunicações e financeiro.
    Conhecimento em Oracle e SQL Server.
    Conhecimento em Machine learning e Inteligência Artificial. e profundo conhecimento em engenheria de dados com machine learning e Inteligência Artificial.
    Ingles nível intermediário.
    Espanhol nível intermediário.
    Certificado em AWS Cloud practitioner 
    """
    embedding = model.encode([cv_text])[0]

    # Busca vagas mais similares
    indices = annoy_index.get_nns_by_vector(embedding, 5, include_distances=False)
    recomendacoes = df.iloc[indices][["titulo_vaga", "competencia_tecnicas_e_comportamentais", "areas_atuacao_vaga"]]
    print("\nTop 5 recomendações de vaga:")
    print(recomendacoes)
