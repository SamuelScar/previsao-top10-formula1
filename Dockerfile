FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements.txt

COPY src/ ./src/
COPY docker/entrypoint.sh /usr/local/bin/tp1

RUN chmod +x /usr/local/bin/tp1 \
    && mkdir -p dados logs resultados \
    && chmod 0777 dados logs resultados

ENTRYPOINT ["tp1"]
CMD ["verificar"]
