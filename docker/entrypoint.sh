#!/bin/sh

set -eu

command_name="${1:-verificar}"
shift || true

case "$command_name" in
    verificar)
        exec python src/treinamento/treinar_modelos.py --stage-check "$@"
        ;;
    preprocessar)
        exec python src/preprocessamento/preparar_recorte.py "$@"
        ;;
    validar)
        exec python src/treinamento/treinar_modelos.py "$@"
        ;;
    testar)
        exec python src/treinamento/treinar_modelos.py --stage-final "$@"
        ;;
    analisar)
        exec python src/analise/analisar_resultados.py "$@"
        ;;
    ajuda|--help|-h)
        echo "Uso: docker compose run --rm tp1 <comando> [opções]"
        echo
        echo "Comandos:"
        echo "  verificar     valida o ambiente e a base sem treinar"
        echo "  preprocessar  recria a base de checkpoints"
        echo "  validar       executa a validação temporal"
        echo "  testar        executa o teste final de 2020"
        echo "  analisar      gera o relatório e os gráficos"
        ;;
    *)
        echo "Comando desconhecido: $command_name" >&2
        echo "Use 'ajuda' para listar os comandos disponíveis." >&2
        exit 2
        ;;
esac
