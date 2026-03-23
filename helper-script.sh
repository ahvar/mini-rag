#!/usr/bin/env bash

# Source this file to add the project root to PYTHONPATH.
# This lets imports like `app.libs.openai_py.openai` and
# `app.libs.pinecone_py.pinecone` resolve from this repo first.

# Resolve the directory where this script lives.
_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"


# Add the project root and the directory containing scrape_and_vectorize_content.py to PYTHONPATH.
SCRIPT_PATH="${_HELPER_DIR}/app/scripts/scrape_and_vectorize_content.py"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${_HELPER_DIR}:$SCRIPT_DIR:${PYTHONPATH}"
else
  export PYTHONPATH="${_HELPER_DIR}:$SCRIPT_DIR"
fi

unset _HELPER_DIR
