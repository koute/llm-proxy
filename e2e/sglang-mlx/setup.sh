#!/usr/bin/env bash
set -euo pipefail

project=$(dirname "$(realpath "$0")")
patches=$(dirname "$(dirname "$project")")/patches
source=$project/sglang

if [[ ! -d $source ]]; then
    git clone --depth 1 --branch v0.5.20 https://github.com/sgl-project/sglang.git "$source"
    mv "$source/python/pyproject_other.toml" "$source/python/pyproject.toml"
fi

uv sync --project "$project" --reinstall-package sglang
site_packages=$("$project/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cp "$patches/llm_proxy_qos.py" "$site_packages/"
patch -p1 -d "$site_packages" < "$patches/sglang-0.5.20.patch"
