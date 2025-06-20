FROM python:3.10-slim


# Instalar gcc e as dependências de compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# Cria o diretório de trabalho
WORKDIR /app

# Copia os arquivos de requirements e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código para o container
COPY . .

# Define a porta (se necessário) e o entrypoint

CMD ["python", "main.py"]