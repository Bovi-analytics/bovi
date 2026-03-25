#!/usr/bin/env bash
# Load Pulumi environment variables from .env files.
# Source this script from justfile recipes:  source scripts/load-env.sh
set -euo pipefail

if [[ ! -f .env ]]; then
    echo "Error: .env file not found! Run 'just configure' first."
    exit 1
fi

# Load env vars first — Pulumi needs AZURE_STORAGE_ACCOUNT/KEY to reach its state backend
set -a
source .env
set +a

CURRENT_STACK=$(pulumi stack --show-name)

# Layer stack-specific overrides if they exist
set -a
[[ -f ".env.${CURRENT_STACK}" ]] && source ".env.${CURRENT_STACK}"
set +a

export PULUMI_CONFIG_PASSPHRASE
