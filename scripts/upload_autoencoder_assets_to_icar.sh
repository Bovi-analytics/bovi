#!/usr/bin/env bash
set -euo pipefail

model_version="${AUTOENCODER_MODEL_VERSION:-v15}"
prefix="${AUTOENCODER_MODEL_PREFIX:-data/models/lactation_autoencoder/versions/${model_version}}"
source_root="${1:-.}"

required=(
  "config/config.yaml"
  "inputs/inference/pkl/event_to_idx_dict.pkl"
  "weights/autoencoder/saved_model.pb"
  "weights/autoencoder/variables/variables.index"
)

for var in STORAGE_ACCOUNT_NAME_ICAR STORAGE_ACCOUNT_KEY_ICAR STORAGE_ACCOUNT_CONTAINER_ICAR; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: ${var} is required." >&2
    exit 1
  fi
done

for suffix in "${required[@]}"; do
  if [ ! -f "${source_root}/${prefix}/${suffix}" ]; then
    echo "ERROR: required autoencoder asset missing: ${source_root}/${prefix}/${suffix}" >&2
    exit 1
  fi
done

if ! compgen -G "${source_root}/${prefix}/weights/autoencoder/variables/variables.data-*" >/dev/null; then
  echo "ERROR: required autoencoder variables.data-* asset missing under ${source_root}/${prefix}/weights/autoencoder/variables" >&2
  exit 1
fi

az storage blob upload-batch \
  --account-name "${STORAGE_ACCOUNT_NAME_ICAR}" \
  --account-key "${STORAGE_ACCOUNT_KEY_ICAR}" \
  --destination "${STORAGE_ACCOUNT_CONTAINER_ICAR}" \
  --source "${source_root}" \
  --pattern "${prefix}/*" \
  --overwrite true
