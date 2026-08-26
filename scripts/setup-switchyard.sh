#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust/Cargo is required for the official standalone Switchyard server."
  echo "Install Rust from https://rustup.rs, restart the shell, then rerun this script."
  exit 1
fi

# Pin the reviewed upstream revision because Switchyard is explicitly pre-alpha.
# This revision includes Stage Router decision-source projection in /v1/stats.
switchyard_rev="${SWITCHYARD_GIT_REV:-574e0cdb1016923091ecf4d15e458ee922d0f189}"
cargo install --locked --force \
  --git https://github.com/NVIDIA-NeMo/Switchyard.git \
  --rev "$switchyard_rev" switchyard-server

server_bin="$(command -v switchyard-server 2>/dev/null || true)"
if [[ -z "$server_bin" && -x "$HOME/.cargo/bin/switchyard-server" ]]; then
  server_bin="$HOME/.cargo/bin/switchyard-server"
fi

"$server_bin" --help >/dev/null
echo "Installed: $("$server_bin" --version)"
echo "Validate with: $server_bin --config config/switchyard/routes.toml --dry-run"
