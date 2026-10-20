#!/usr/bin/env bash
set -euo pipefail

engine=$1
shift
e2e_directory=$(dirname "$(realpath "$0")")
patches_directory=$(dirname "$e2e_directory")/patches
qos_module=$patches_directory/llm_proxy_qos.py
engine_patch=$(find "$patches_directory" -name "$engine-*.patch")
venv=/tmp/llm-proxy-e2e/$engine
venv_inputs_hash=$(cat "$qos_module" "$engine_patch" "$e2e_directory/$engine"/{pyproject.toml,uv.lock} | sha256sum)

if [[ "$(cat "$venv/inputs.sha256" 2>/dev/null)" != "$venv_inputs_hash" ]]; then
    rm -rf "$venv"
    UV_PROJECT_ENVIRONMENT=$venv UV_LINK_MODE=copy uv sync --locked --project "$e2e_directory/$engine"
    site_packages=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    cp "$qos_module" "$site_packages/"
    patch -p1 -d "$site_packages" < "$engine_patch"
    echo "$venv_inputs_hash" > "$venv/inputs.sha256"
fi

exec "$venv/bin/python" -m pytest "$e2e_directory" --engine "$engine" "$@"
