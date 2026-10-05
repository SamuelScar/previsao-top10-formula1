# Previsão do top 10 na Fórmula 1

O projeto investiga em qual momento de uma corrida já é possível prever quais
pilotos ativos terminarão no top 10. O pré-processamento está concluído e o
treinamento temporal foi executado com resultados em `resultados/treinamento/`.

O relatório completo do projeto está disponível em
[`previsao-top-10-formula-1.pdf`](previsao-top-10-formula-1.pdf).

Todo o pipeline Python é executado com Docker. Não é necessário criar ambiente
virtual nem instalar as dependências no computador.

## Relatório

- [`previsao-top-10-formula-1.pdf`](previsao-top-10-formula-1.pdf): relatório
  final com a metodologia, os experimentos e os resultados.

## Execução com Docker

Pré-requisitos:

- Docker Engine;
- Docker Compose v2.

Na raiz do projeto, construa a imagem:

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
