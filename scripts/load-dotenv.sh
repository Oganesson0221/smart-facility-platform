#!/usr/bin/env bash

load_dotenv_file() {
  local env_file="${1:-.env}"
  if [[ ! -f "$env_file" ]]; then
    return 0
  fi

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    raw_line="${raw_line%$'\r'}"
    [[ -z "$raw_line" ]] && continue
    [[ "$raw_line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$raw_line" != *"="* ]]; then
      continue
    fi

    local key="${raw_line%%=*}"
    local value="${raw_line#*=}"

    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"

    if [[ -z "$key" ]]; then
      continue
    fi

    export "$key=$value"
  done < "$env_file"
}
