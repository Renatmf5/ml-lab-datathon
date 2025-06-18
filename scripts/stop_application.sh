#!/bin/bash
echo "Executando Application Stop: encerrando a aplicação de forma graciosa..."
# Envia sinal SIGTERM para o processo da aplicação. Ajuste o critério de busca se necessário.
pkill -SIGTERM -f main.py || true