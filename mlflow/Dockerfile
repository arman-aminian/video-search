FROM ghcr.io/mlflow/mlflow:v2.5.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install 'psycopg2<2.9'
