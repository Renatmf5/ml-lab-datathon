from pipelines.training_pipeline import run_training_pipeline
from features.build_features import build
from models.drift_detector import main

if __name__ == '__main__':
    print("Iniciando build das features...")
    build()
    print("Build das features finalizado.")
    
    print("Iniciando pipeline de treinamento...")
    run_training_pipeline()
    
    print("fazendo drift detector...")
    main()  # Chama a função principal do drift detector
    print("Drift detector finalizado.")