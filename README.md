# TP1 — Análise preditiva de Fórmula 1

O projeto investiga em qual momento de uma corrida já é possível prever quais
pilotos ativos terminarão no top 10. O pré-processamento está concluído e o
treinamento temporal foi executado com resultados em `resultados/treinamento/`.

O relatório acadêmico consolidado está disponível em
[`previsao-top-10-formula-1.pdf`](previsao-top-10-formula-1.pdf).

Todo o pipeline Python é executado com Docker. Não é necessário criar ambiente
virtual nem instalar as dependências no computador.

## Estrutura

```text
.
├── dados/          # dados originais e base processada
├── docker/         # entrada dos comandos executados no contêiner
├── logs/           # logs gerados pelas execuções
├── resultados/     # métricas, previsões, relatório e gráficos
├── src/            # pré-processamento, treinamento e análise
├── compose.yaml
├── Dockerfile
├── previsao-top-10-formula-1.pdf
└── requirements.txt
```

## Fluxo ativo

- `dados/originais/`: CSVs de origem preservados;
- `src/preprocessamento/preparar_recorte.py`: gera a base de checkpoints;
- `dados/gerados/base_checkpoints.csv`: entrada pronta para o treinamento;
- `src/treinamento/treinar_modelos.py`: valida e compara os cinco
  classificadores;
- `resultados/treinamento/`: métricas e previsões produzidas pelo treinamento;
- `logs/`: auditoria separada por etapa;
- `previsao-top-10-formula-1.pdf`: relatório acadêmico final.

Os materiais fornecidos pelo professor ficam fora deste repositório, no
diretório local `../materiais-professor/`.

## Relatório

- [`previsao-top-10-formula-1.pdf`](previsao-top-10-formula-1.pdf): relatório
  final do Trabalho Prático 1.

## Execução com Docker

Pré-requisitos:

- Docker Engine;
- Docker Compose v2.

Na raiz do TP1, construa a imagem:

```bash
docker compose build
```

Confira o ambiente e a base sem treinar:

```bash
docker compose run --rm tp1 verificar
```

Para recriar a base de checkpoints:

```bash
docker compose run --rm tp1 preprocessar --overwrite
```

Execute a validação temporal de 2018 e 2019:

```bash
docker compose run --rm tp1 validar
```

Somente depois, execute o teste final de 2020:

```bash
docker compose run --rm tp1 testar
```

Gere o relatório de análise e os gráficos:

```bash
docker compose run --rm tp1 analisar
```

Use `--overwrite` nos comandos que precisarem substituir saídas existentes. Os
diretórios `dados/`, `logs/` e `resultados/` são montados no contêiner, portanto
os arquivos gerados permanecem no projeto após o encerramento da execução.

Para listar os comandos disponíveis:

```bash
docker compose run --rm tp1 ajuda
```

Se o usuário do computador não utilizar UID e GID `1000`, informe-os antes do
comando para que os arquivos sejam criados com as permissões corretas:

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm tp1 verificar
```
