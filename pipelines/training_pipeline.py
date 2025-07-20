from models.Modelo_Matching_Classificacao import prepare_matching_data 
from models.Modelo_Matching_Classificacao import train_model_tfidVectorizer
from models.Modelo_Recomendacao_Vagas import prepare_recommendation_data
from models.Modelo_Recomendacao_Vagas import train_model_recommendation



def run_training_pipeline():
    # Primeiro, prepare os dados de matching
    print("Preparando dados de matching...")
    prepare_matching_data.build()  # supondo que a função build() 

    # Em seguida, treine o modelo
    print("Iniciando treinamento do modelo de matching...")
    model_version = train_model_tfidVectorizer.run()
    
    # Agora, preparo os dados de recomendação de vagas
    print("Preparando dados de recomendação de vagas...")
    prepare_recommendation_data.build()  # supondo que a função build()
    #model_version = "v5"  # Defina a versão do modelo conforme necessário
    print("Treinamento do modelo de recomendação de vagas iniciado...")
    train_model_recommendation.run(model_version)  # inicia o treinamento do modelo de recomendação
    print("Pipeline de treinamento concluído com sucesso.")