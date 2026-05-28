#!/usr/bin/env bash
# Load Pulumi environment variables from .env files.
# Source this script from justfile recipes:  source scripts/load-env.sh
set -euo pipefail

if [[ ! -f .env ]]; then
    echo "Error: .env file not found! Run 'just configure' first."
    exit 1
fi

# Load env vars first — Pulumi needs AZURE_STORAGE_ACCOUNT_STATE/KEY to reach its state backend
set -a
source .env
set +a

if [[ -n "${AZURE_STORAGE_ACCOUNT_STATE:-}" ]]; then
    export AZURE_STORAGE_ACCOUNT="$AZURE_STORAGE_ACCOUNT_STATE"
fi
if [[ -n "${AZURE_STORAGE_KEY_STATE:-}" ]]; then
    export AZURE_STORAGE_KEY="$AZURE_STORAGE_KEY_STATE"
fi
if [[ -n "${PULUMI_PASSPHRASE:-}" ]]; then
    export PULUMI_CONFIG_PASSPHRASE="$PULUMI_PASSPHRASE"
fi

CURRENT_STACK=$(pulumi stack --show-name)

# Layer stack-specific overrides if they exist
set -a
[[ -f ".env.${CURRENT_STACK}" ]] && source ".env.${CURRENT_STACK}"
set +a

if [[ -n "${PULUMI_PASSPHRASE:-}" ]]; then
    export PULUMI_CONFIG_PASSPHRASE="$PULUMI_PASSPHRASE"
fi
export PULUMI_CONFIG_PASSPHRASE
