from pipelines.training_pipeline import run_training_pipeline
from features.build_features import build

if __name__ == '__main__':
    print("Iniciando build das features...")
    build()
    print("Build das features finalizado.")
    
    print("Iniciando pipeline de treinamento...")
    run_training_pipeline()