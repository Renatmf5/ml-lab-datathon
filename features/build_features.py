import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from data import load_applicants, load_prospects, load_vagas, save_features
import re

bucket = "decision-data-lake"  # Alterar conforme necessário
FEATURES_PATH = Path("data/features")
FEATURES_PATH.mkdir(parents=True, exist_ok=True)

# Dicionário mapeando DDD para "Cidade, Estado"
DDD_TO_LOCATION = {
    "68": "Rio Branco, Acre",
    "96": "Macapá, Amapá",
    "92": "Manaus, Amazonas",
    "97": "Tefé/Coari, Amazonas",
    "69": "Porto Velho, Rondônia",
    "95": "Boa Vista, Roraima",
    "63": "Palmas, Tocantins",
    "82": "Maceió, Alagoas",
    "71": "Salvador, Bahia",
    "73": "Itabuna/Ilhéus, Bahia",
    "74": "Juazeiro, Bahia",
    "75": "Feira de Santana, Bahia",
    "77": "Vitória da Conquista, Bahia",
    "85": "Fortaleza, Ceará",
    "88": "Juazeiro do Norte, Ceará",
    "98": "São Luís, Maranhão",
    "99": "Imperatriz, Maranhão",
    "83": "João Pessoa, Paraíba",
    "81": "Recife, Pernambuco",
    "87": "Petrolina, Pernambuco",
    "86": "Teresina, Piauí",
    "89": "Picos/Floriano, Piauí",
    "84": "Natal, Rio Grande do Norte",
    "79": "Aracaju, Sergipe",
    "61": "Brasília, Distrito Federal",
    "62": "Goiânia, Goiás",
    "64": "Rio Verde, Goiás",
    "65": "Cuiabá, Mato Grosso",
    "66": "Rondonópolis, Mato Grosso",
    "67": "Campo Grande, Mato Grosso do Sul",
    "27": "Vitória, Espírito Santo",
    "28": "Cachoeiro de Itapemirim, Espírito Santo",
    "31": "Belo Horizonte, Minas Gerais",
    "32": "Juiz de Fora, Minas Gerais",
    "33": "Governador Valadares, Minas Gerais",
    "34": "Uberlândia, Minas Gerais",
    "35": "Poços de Caldas, Minas Gerais",
    "37": "Divinópolis, Minas Gerais",
    "38": "Montes Claros, Minas Gerais",
    "21": "Rio de Janeiro, Rio de Janeiro",
    "22": "Campos dos Goytacazes, Rio de Janeiro",
    "24": "Volta Redonda, Rio de Janeiro",
    "11": "São Paulo, São Paulo",
    "12": "São José dos Campos, São Paulo",
    "13": "Santos, São Paulo",
    "14": "Bauru, São Paulo",
    "15": "Sorocaba, São Paulo",
    "16": "Ribeirão Preto, São Paulo",
    "17": "São José do Rio Preto, São Paulo",
    "18": "Presidente Prudente, São Paulo",
    "19": "Campinas, São Paulo",
    "41": "Curitiba, Paraná",
    "42": "Ponta Grossa, Paraná",
    "43": "Londrina, Paraná",
    "44": "Maringá, Paraná",
    "45": "Cascavel, Paraná",
    "46": "Francisco Beltrão, Paraná",
    "51": "Porto Alegre, Rio Grande do Sul",
    "53": "Pelotas, Rio Grande do Sul",
    "54": "Caxias do Sul, Rio Grande do Sul",
    "55": "Santa Maria, Rio Grande do Sul",
    "47": "Joinville, Santa Catarina",
    "48": "Florianópolis, Santa Catarina",
    "49": "Chapecó, Santa Catarina"
}

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%d-%m-%Y")
    except:
        return None

def get_location_by_ddd(phone: str) -> str:
    """
    Extrai o DDD do telefone e retorna a localização (cidade, estado)
    a partir do mapeamento em DDD_TO_LOCATION.
    """
    digits = re.sub(r"\D", "", phone)
    if len(digits) >= 2:
        ddd = digits[:2]
        return DDD_TO_LOCATION.get(ddd)
    return None

def fill_location(phone: str, current_location: str) -> str:
    """
    Se current_location estiver vazio, tenta preencher com a localização
    obtida a partir do DDD do telefone.
    """
    if current_location and current_location.strip():
        return current_location
    location = get_location_by_ddd(phone)
    if location:
        return location
    return current_location

def fill_value(value):
    """
    Retorna "Não Informado" se o valor for None, string vazia ou um hífen.
    Caso contrário, retorna o valor original.
    """
    if value in [None, '', '-']:
        return "Não Informado"
    return value

def fill_row(row: dict, skip_keys=None) -> dict:
    """
    Percorre os campos do dicionário e preenche os valores ausentes com "Não Informado",
    exceto as chaves especificadas em skip_keys.
    """
    if skip_keys is None:
        skip_keys = []
    return {k: (fill_value(v) if k not in skip_keys else v) for k, v in row.items()}

def build():
    prospect_data = load_prospects(bucket)
    applicants_data = load_applicants(bucket)
    vagas_data = load_vagas(bucket)

    rows = []
    for vaga_id, vaga in prospect_data.items():
        for prospect in vaga["prospects"]:
            codigo = prospect["codigo"]
            vagas = vagas_data.get(vaga_id, {})

            vagas_info = vagas.get("informacoes_basicas", {})
            vagas_perfil = vagas.get("perfil_vaga", {})
            vagas_beneficios = vagas.get("beneficios", {})

            applicant = applicants_data.get(codigo, {})
            applicant_info = applicant.get("infos_basicas", {})
            pessoal = applicant.get("informacoes_pessoais", {})
            applicant_profissional = applicant.get("informacoes_profissionais", {})
            formacao = applicant.get("formacao_e_idiomas", {})

            row = {
                # Informações básicas pessoais do candidato
                "codigo": codigo,
                "nome": prospect["nome"],
                "situacao_candidato": prospect.get("situacao_candidado"),
                # Preenche cidade a partir do telefone se necessário
                "cidade": fill_location(applicant_info.get("telefone", ""), applicant_info.get("local", "")),
                "sexo": pessoal.get("sexo"),
                "estado_civil": pessoal.get("estado_civil"),
                "pcd": pessoal.get("pcd"),
                "data_candidatura": parse_date(prospect["data_candidatura"]),
                "ultima_atualizacao": parse_date(prospect["ultima_atualizacao"]),
                "dias_entre_candidatura_e_atualizacao": None,
                "sabendo_vaga_por": applicant_info.get("sabendo_de_nos_por"),
                "comentario": prospect.get("comentario"),

                # Informações profissionais do candidato
                "titulo_profissional": applicant_profissional.get("titulo_profissional"),
                "area_atuacao": applicant_profissional.get("area_atuacao"),
                "conhecimentos_tecnicos": applicant_profissional.get("conhecimentos_tecnicos"),
                "certificacoes": applicant_profissional.get("certificacoes"),
                "outras_certificacoes": applicant_profissional.get("outras_certificacoes"),
                "remuneracao": applicant_profissional.get("remuneracao"),
                "nivel_profissional": applicant_profissional.get("nivel_profissional"),

                # Informações acadêmicas e de idiomas do candidato
                "nivel_academico": formacao.get("nivel_academico"),
                "instituicao_ensino_superior": formacao.get("instituicao_ensino_superior"),
                "cursos": formacao.get("cursos"),
                "ingles": formacao.get("nivel_ingles"),
                "espanhol": formacao.get("nivel_espanhol"),
                "outros_idiomas": formacao.get("outro_idioma"),

                # Informações sobre a vaga
                "titulo_vaga": vagas_info.get("titulo_vaga"),
                "vaga_sap": vagas_info.get("vaga_sap"),
                "tipo_contratacao": vagas_info.get("tipo_contratacao"),
                "prazo_contratacao": vagas_info.get("prazo_contratacao"),
                "prioridade_vaga": vagas_info.get("prioridade_vaga"),
                "objetivo_vaga": vagas_info.get("objetivo_vaga"),
                "pais_vaga": vagas_perfil.get("pais"),
                "estado_vaga": vagas_perfil.get("estado"),
                "cidade_vaga": vagas_perfil.get("cidade"),
                "vaga_especifica_para_pcd": vagas_perfil.get("vaga_especifica_para_pcd"),
                "nivel_profissional_vaga": vagas_perfil.get("nivel profissional"),
                "nivel_academico_vaga": vagas_perfil.get("nivel_academico"),
                "nivel_ingles_vaga": vagas_perfil.get("nivel_ingles"),
                "nivel_espanhol_vaga": vagas_perfil.get("nivel_espanhol"),
                "outros_idiomas_vaga": vagas_perfil.get("outro_idioma"),
                "areas_atuacao_vaga": vagas_perfil.get("areas_atuacao"),
                "principais_atividades": vagas_perfil.get("principais_atividades"),
                "competencia_tecnicas_e_comportamentais": vagas_perfil.get("competencia_tecnicas_e_comportamentais"),
                "demais_observacoes": vagas_perfil.get("demais_observacoes"),
                "viagens_requeridas": vagas_perfil.get("viagens_requeridas"),

                # Informações de benefícios da vaga
                "valor_venda": vagas_beneficios.get("valor_venda"),
                "valor_compra_1": vagas_beneficios.get("valor_compra_1"),
                "valor_compra_2": vagas_beneficios.get("valor_compra_2"),

                # Informações do CV do candidato
                "cv_candidato": applicant.get("cv_pt"),
            }

            # Calcula a diferença em dias, se possível
            if row["data_candidatura"] and row["ultima_atualizacao"]:
                row["dias_entre_candidatura_e_atualizacao"] = (
                    row["ultima_atualizacao"] - row["data_candidatura"]
                ).days

            # Converte os campos de data para string (tipo único para o DataFrame)
            row["data_candidatura"] = row["data_candidatura"].isoformat() if row["data_candidatura"] else "Não Informado"
            row["ultima_atualizacao"] = row["ultima_atualizacao"].isoformat() if row["ultima_atualizacao"] else "Não Informado"
            row["dias_entre_candidatura_e_atualizacao"] = (
                str(row["dias_entre_candidatura_e_atualizacao"]) if row["dias_entre_candidatura_e_atualizacao"] is not None else "Não Informado"
            )

            # Aplica a limpeza para outros campos (exceto os já convertidos)  
            skip_keys = ["data_candidatura", "ultima_atualizacao", "dias_entre_candidatura_e_atualizacao"]
            row = fill_row(row, skip_keys=skip_keys)
            
            # --- Início das melhorias para enriquecer a semântica dos campos ---
            # Para candidato:
            if row.get("ingles"):
                row["ingles"] = f"Inglês nível {row['ingles']}"
            if row.get("espanhol"):
                row["espanhol"] = f"Espanhol nível {row['espanhol']}"
            if row.get("outros_idiomas"):
                row["outros_idiomas"] = f"Outros idiomas: {row['outros_idiomas']}"
            if row.get("cidade"):
                row["cidade"] = f"Cidade e estado onde o candidato reside: {row['cidade']}"
            if row.get("pcd") and row["pcd"] != "Não Informado":
                row["pcd"] = f"{row['pcd']} é PCD"
            
            # Para vaga:
            if row.get("nivel_profissional_vaga"):
                row["nivel_profissional_vaga"] = f"Nível profissional da vaga: {row['nivel_profissional_vaga']}"
            if row.get("cidade_vaga"):
                row["cidade_vaga"] = f"Cidade da vaga: {row['cidade_vaga']}"
            if row.get("estado_vaga"):
                row["estado_vaga"] = f"Estado da vaga: {row['estado_vaga']}"
            if row.get("vaga_especifica_para_pcd") and row["vaga_especifica_para_pcd"] != "Não Informado":
                row["vaga_especifica_para_pcd"] = f"{row['vaga_especifica_para_pcd']} é vaga PCD"
            # --- Fim das melhorias ---
            rows.append(row)

    df = pd.DataFrame(rows)
    
    local_file = FEATURES_PATH / "candidates.parquet"
    df.to_parquet(local_file, index=False)
    print(f"Features salvas localmente em {local_file}")
    
    # Upload das features para o S3
    save_features(df, bucket, "candidates")
    print(f"Features salvas em s3://{bucket}/features/candidates.parquet")
