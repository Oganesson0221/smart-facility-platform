#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

find_switchyard_server() {
  local candidate
  for candidate in \
    "${SWITCHYARD_SERVER_BIN:-}" \
    "$(command -v switchyard-server 2>/dev/null || true)" \
    "$HOME/.cargo/bin/switchyard-server"
  do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

server_bin="$(find_switchyard_server || true)"
if [[ -z "${server_bin:-}" ]]; then
  echo "switchyard-server was not found."
  echo "Install it with: ./scripts/setup-switchyard.sh"
  exit 1
fi

config_path="${SWITCHYARD_CONFIG:-config/switchyard/routes.toml}"
host="${SWITCHYARD_HOST:-0.0.0.0}"
port="${SWITCHYARD_PORT:-4000}"
routing_log="${SWITCHYARD_ROUTING_LOG_PATH:-logs/switchyard-routing.jsonl}"

if [[ ! -f "$config_path" ]]; then
  echo "Switchyard configuration not found: $config_path"
  exit 1
fi

mkdir -p "$(dirname "$routing_log")"
"$server_bin" --config "$config_path" --dry-run

echo "Starting Switchyard Stage Router on port ${port}."
echo "Route: ${SWITCHYARD_MODEL:-switchyard/exitwatch-stage}"
echo "Routing log: ${routing_log}"
exec "$server_bin" \
  --config "$config_path" \
  --host "$host" \
  --port "$port" \
  --routing-log-file "$routing_log"
