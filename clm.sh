#!/usr/bin/env bash
set -Eeuo pipefail

# Controle manual dos serviços CLM na Jetson.
# Uso: ./clm.sh {start|stop|restart|status|logs|logs-clm|logs-encoder|down|pull}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.jetson.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Erro: não encontrei o arquivo Compose: $COMPOSE_FILE" >&2
  echo "Coloque este script na raiz do repositório CLM." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Erro: Docker não está instalado ou não está no PATH." >&2
  exit 1
fi

# Se o usuário não tiver permissão para acessar o daemon, oriente sem elevar
# privilégios silenciosamente.
if ! docker info >/dev/null 2>&1; then
  echo "Sem acesso ao daemon Docker para o usuário atual." >&2
  echo "Execute com sudo: sudo $0 $*" >&2
  exit 1
fi

compose() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

usage() {
  echo "Uso: $0 {start|stop|restart|status|logs|logs-clm|logs-encoder|down|pull}"
}

case "${1:-}" in
  start)
    # Inicia os serviços apenas quando solicitado.
    compose up -d
    compose ps
    ;;
  stop)
    # Para os serviços, mantendo containers e configuração para uso posterior.
    compose stop
    ;;
  restart)
    compose restart
    ;;
  status|ps)
    compose ps
    ;;
  logs)
    compose logs -f
    ;;
  logs-clm)
    compose logs -f clm
    ;;
  logs-encoder)
    compose logs -f encoder
    ;;
  down)
    # Remove os containers e a rede Compose; imagens e caches/modelos persistem.
    compose down
    ;;
  pull)
    compose pull
    ;;
  *)
    usage
    exit 2
    ;;
esac
