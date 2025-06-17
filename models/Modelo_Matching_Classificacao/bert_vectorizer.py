# filepath: bert_vectorizer.py
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import logging

class BERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2", batch_size=32, device="cpu", shared_model=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.shared_model = shared_model
    
    def fit(self, X, y=None):
        if self.shared_model is not None:
            self.model_ = self.shared_model
            logging.info(f"Utilizando modelo compartilhado {self.model_name} na {self.device}")
        else:
            logging.info(f"Carregando o modelo {self.model_name} na {self.device}")
            self.model_ = SentenceTransformer(self.model_name, device=self.device)
        return self
    
    def transform(self, X):
        texts = X.tolist() if hasattr(X, "tolist") else list(X)
        logging.info(f"Transformando {len(texts)} textos em embeddings")
        embeddings = self.model_.encode(texts, batch_size=self.batch_size, show_progress_bar=True)
        return embeddings