#!/bin/bash
echo "Executando Application Start: iniciando a aplicação..."
# Inicia a aplicação – se for o entrypoint do container, cuidado para não criar processos duplicados.
exec python main.py