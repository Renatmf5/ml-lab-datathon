import pandas as pd
import numpy as np
import logging
import boto3
import io
from datetime import datetime
from alibi_detect.cd import KSDrift
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from stop_words import get_stop_words

# Configurações
BUCKET = "decision-data-lake"                          # Bucket onde o relatório será salvo
BUCKET_FEATURES = "decision-data-lake-features"        # Bucket versionado com os candidates
KEY = "features/candidates.parquet"                   # Caminho do arquivo no bucket versionado
S3_OUTPUT_KEY = "drift_reports/drift_report.html"

# Combina stop word em portugues e ingles
pt_stop_words = set(get_stop_words('portuguese'))
combined_stop_words = list(pt_stop_words.union(ENGLISH_STOP_WORDS))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_parquet_from_s3(bucket: str, key: str, version_id: str = None) -> pd.DataFrame:
    """Carrega um arquivo parquet do S3, com suporte a versionamento."""
    s3 = boto3.client("s3")
    params = {"Bucket": bucket, "Key": key}
    if version_id:
        params["VersionId"] = version_id
    try:
        obj = s3.get_object(**params)
        parquet_buffer = io.BytesIO(obj["Body"].read())
        logging.info(f"Carregado com sucesso s3://{bucket}/{key}" + (f" (VersionId: {version_id})" if version_id else ""))
        return pd.read_parquet(parquet_buffer)
    except Exception as e:
        logging.error(f"Erro ao carregar s3://{bucket}/{key}: {e}")
        raise

def get_versions(bucket: str, key: str) -> list:
    """Retorna a lista de versões do objeto no S3, ordenadas da mais recente para a mais antiga."""
    s3 = boto3.client("s3")
    try:
        response = s3.list_object_versions(Bucket=bucket, Prefix=key)
        versions = [v for v in response.get("Versions", []) if v['Key'] == key]
        if not versions:
            return []
        versions_sorted = sorted(versions, key=lambda v: v["LastModified"], reverse=True)
        return versions_sorted
    except Exception as e:
        logging.error(f"Erro ao listar versões para s3://{bucket}/{key}: {e}")
        raise

def get_penultimate_version_id(versions: list) -> str:
    """Retorna o VersionId da penúltima versão (índice 1) da lista ordenada."""
    if len(versions) < 2:
        raise ValueError("Versões insuficientes para comparação de curto prazo.")
    return versions[1]["VersionId"]

def get_oldest_version_id(versions: list) -> str:
    """Retorna o VersionId da versão mais antiga (último elemento) da lista."""
    if not versions:
        raise ValueError("Nenhuma versão encontrada.")
    return versions[-1]["VersionId"]

def upload_report_to_s3(report_html: str, bucket: str, key: str):
    """Envia o relatório HTML para o S3."""
    s3 = boto3.client("s3")
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=report_html, ContentType="text/html")
        logging.info(f"Relatório de drift enviado para s3://{bucket}/{key}")
    except Exception as e:
        logging.error(f"Erro ao enviar relatório para o S3: {e}")
        raise

def calculate_psi(baseline_values: np.ndarray, current_values: np.ndarray, buckets: int = 10) -> float:
    """
    Calcula o Population Stability Index (PSI) para uma única feature (vetor).
    """
    epsilon = 1e-4  # valor pequeno para evitar divisão por zero
    # Define os limites dos bins com base nos dados combinados
    combined = np.concatenate([baseline_values, current_values])
    if combined.min() == combined.max():
        bins = np.array([combined.min(), combined.max() + 1e-4])
    else:
        bins = np.linspace(combined.min(), combined.max(), buckets + 1)
    
    baseline_hist, _ = np.histogram(baseline_values, bins=bins)
    current_hist, _ = np.histogram(current_values, bins=bins)
    
    baseline_perc = baseline_hist / baseline_hist.sum() if baseline_hist.sum() > 0 else np.zeros_like(baseline_hist)
    current_perc = current_hist / current_hist.sum() if current_hist.sum() > 0 else np.zeros_like(current_hist)

    psi = 0.0
    for bp, cp in zip(baseline_perc, current_perc):
        # Ajusta valores zero
        bp = bp if bp > 0 else epsilon
        cp = cp if cp > 0 else epsilon
        psi += (cp - bp) * np.log(cp / bp)
    return psi

def run_ks_drift_enhanced(baseline: pd.DataFrame, current: pd.DataFrame, p_val: float = 0.05, top_n: int = 10) -> dict:
    """
    Executa o detector de drift KSDrift nas representações numéricas extraídas via TfidfVectorizer.
    Além disso, calcula a diferença média para cada termo entre o baseline e o conjunto atual, bem como o PSI de cada termo.
    Agora também calcula o PSI global, considerando todo o conjunto TF-IDF (após flatten).
    """
    # Concatena as colunas textuais em uma única string por registro
    baseline_text = baseline.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    current_text = current.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    
    # Cria a representação TF-IDF baseada no baseline e transforma o conjunto atual
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=combined_stop_words)
    baseline_vect = vectorizer.fit_transform(baseline_text).toarray()
    current_vect = vectorizer.transform(current_text).toarray()
    
    # Executa o KSDrift no espaço numérico
    cd = KSDrift(baseline_vect, p_val=p_val)
    ks_result = cd.predict(current_vect)
    
    # Calcula a média de cada termo, a diferença e o PSI individual
    baseline_avg = np.mean(baseline_vect, axis=0)
    current_avg = np.mean(current_vect, axis=0)
    diff = current_avg - baseline_avg
    feature_names = np.array(vectorizer.get_feature_names_out())

    feature_differences = []
    for i, term in enumerate(feature_names):
        psi = calculate_psi(baseline_vect[:, i], current_vect[:, i])
        feature_differences.append({
            "term": term,
            "baseline_avg": baseline_avg[i],
            "current_avg": current_avg[i],
            "difference": diff[i],
            "psi": psi
        })
    # Seleciona os top_n termos com maior diferença absoluta
    sorted_features = sorted(feature_differences, key=lambda x: abs(x["difference"]), reverse=True)[:top_n]
    
    # PSI global: achatando todos os arrays e calculando o PSI
    global_psi = calculate_psi(baseline_vect.flatten(), current_vect.flatten())
    
    return {"ks_result": ks_result, "top_features": sorted_features, "global_psi": global_psi}

def generate_feature_table_html(top_features: list) -> str:
    """
    Gera uma tabela HTML com os termos que tiveram maiores diferenças entre baseline e dados atuais, incluindo valores de PSI.
    """
    table = """
    <table border="1" cellpadding="5" cellspacing="0">
      <tr>
        <th>Termo</th>
        <th>Média Baseline</th>
        <th>Média Atual</th>
        <th>Diferença</th>
        <th>PSI</th>
      </tr>
    """
    for feat in top_features:
        table += f"""
      <tr>
        <td>{feat['term']}</td>
        <td>{feat['baseline_avg']:.4f}</td>
        <td>{feat['current_avg']:.4f}</td>
        <td>{feat['difference']:.4f}</td>
        <td>{feat['psi']:}</td>
      </tr>
        """
    table += "\n    </table>"
    return table

def generate_html_report(result_short: dict, result_long: dict) -> str:
    """
    Gera um relatório HTML mais rico exibindo os resultados dos testes de drift e, para cada comparação,
    uma tabela com os termos que apresentaram maiores alterações no TF-IDF e seus respectivos valores de PSI.
    Agora inclui também um PSI global.
    """
    def format_result(ks_result: dict) -> str:
        if 'error' in ks_result:
            return ks_result['error']
        is_drift = ks_result['data']['is_drift']
        p_val = ks_result['data']['p_val']
        threshold = ks_result['data']['threshold']
        if isinstance(p_val, np.ndarray):
            p_val = float(np.mean(p_val))
        if isinstance(threshold, np.ndarray):
            threshold = float(np.mean(threshold))
        drift_status = "SIM" if is_drift else "NÃO"
        return f"<p>Drift detectado: <strong>{drift_status}</strong></p>" \
               f"<p>p-value: {p_val:.4f} (limite: {threshold:.4f})</p>"

    html = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Relatório de Data Drift - Alibi Detect</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 20px; }}
          h1 {{ color: #333366; }}
          h2 {{ color: #3366CC; }}
          p, table {{ font-size: 14px; }}
          hr {{ border: 1px solid #cccccc; margin: 30px 0; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ padding: 8px 12px; text-align: left; }}
          th {{ background-color: #f2f2f2; }}
        </style>
      </head>
      <body>
        <h1>Relatório de Data Drift - {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</h1>
        
        <div>
          <h2>Legenda das Métricas</h2>
          <p><strong>KSDrift:</strong> Teste estatístico que compara a distribuição dos dados entre duas amostras utilizando o teste de Kolmogorov-Smirnov para identificar mudanças significativas.</p>
          <p><strong>TF-IDF:</strong> Técnica de extração de características que transforma textos em representações numéricas ponderando a frequência dos termos, destacando aqueles mais informativos.</p>
          <p><strong>PSI (Population Stability Index):</strong> Métrica que avalia a estabilidade dos dados comparando a distribuição dos termos entre duas amostras, indicando a magnitude da mudança ao longo do tempo. PSI global agrega a avaliação de toda a distribuição dos dados.</p>
        </div>

        <h2>Drift de Curto Prazo (Atual vs. Penúltima Versão)</h2>
        {format_result(result_short["ks_result"])}
        <p><strong>PSI Global:</strong> {result_short["global_psi"]:}</p>
        <h3>Top termos com maiores diferenças</h3>
        {generate_feature_table_html(result_short["top_features"])}
        
        <hr>
        
        <h2>Drift de Médio Prazo (Atual vs. Versão Mais Antiga)</h2>
        {format_result(result_long["ks_result"])}
        <p><strong>PSI Global:</strong> {result_long["global_psi"]:.4f}</p>
        <h3>Top termos com maiores diferenças</h3>
        {generate_feature_table_html(result_long["top_features"])}
      </body>
    </html>
    """
    return html

def main():
    logging.info(f"Listando versões para s3://{BUCKET_FEATURES}/{KEY}...")
    versions = get_versions(BUCKET_FEATURES, KEY)
    
    if len(versions) < 2:
        logging.warning("Não há versões suficientes para comparação de drift. Finalizando.")
        return

    # Carrega a versão atual usando a versão mais recente
    logging.info("Carregando a versão atual do arquivo (dados atuais)...")
    current_df = load_parquet_from_s3(BUCKET_FEATURES, KEY, version_id=versions[0]['VersionId'])
        
    # Comparação de curto prazo: penúltima versão
    logging.info("Carregando a penúltima versão (baseline curto prazo)...")
    penultimate_version_id = get_penultimate_version_id(versions)
    baseline_short_df = load_parquet_from_s3(BUCKET_FEATURES, KEY, version_id=penultimate_version_id)
    
    # Comparação de médio prazo: versão mais antiga
    logging.info("Carregando a versão mais antiga (baseline médio prazo)...")
    oldest_version_id = get_oldest_version_id(versions)
    baseline_long_df = load_parquet_from_s3(BUCKET_FEATURES, KEY, version_id=oldest_version_id)
    
    logging.info("Executando detector KSDrift para comparação de curto prazo...")
    result_short = run_ks_drift_enhanced(baseline_short_df, current_df)
    
    logging.info("Executando detector KSDrift para comparação de médio prazo...")
    result_long = run_ks_drift_enhanced(baseline_long_df, current_df)
    
    logging.info("Gerando relatório HTML enriquecido...")
    report_html = generate_html_report(result_short, result_long)
    
    logging.info("Enviando relatório para o S3...")
    upload_report_to_s3(report_html, BUCKET, S3_OUTPUT_KEY)
    
    logging.info("Análise de drift concluída com sucesso.")
    
if __name__ == "__main__":
    main()
