$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"
# Windows sandbox で hardlink/symlink の権限問題を避けるため copy にする。
$env:UV_LINK_MODE = "copy"

. "$PSScriptRoot\_lib.ps1"

Invoke-Checked { uv run ruff format --check python tests/python }
Invoke-Checked { uv run ruff check python tests/python }
Invoke-Checked { uv run ty check python tests/python }
Invoke-Checked { uv run pytest }
